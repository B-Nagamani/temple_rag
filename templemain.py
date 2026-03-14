"""
Temple RAG System - Backend for Streamlit Interface
Handles PDF processing, embeddings, retrieval, and question answering
"""

from pathlib import Path
import textwrap
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# PDF processing
from pypdf import PdfReader

# Embeddings and models
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Vector DB
import chromadb
from chromadb.config import Settings


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for Temple RAG System"""
    
    def __init__(self):
        # Paths
        self.PDF_DIR = "/Users/nagamanibhukya/Downloads/rag sample/Templeslist"
        self.CHROMA_DB_DIR = "chroma_rag_demo"
        
        # Models
        self.EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
        self.GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
        
        # Processing parameters
        self.MAX_PAGES = None  # None = process all pages
        self.CHUNK_MAX_CHARS = 800
        self.CHUNK_OVERLAP_SENTENCES = 1
        self.MAX_CONTEXT_CHARS = 1600
        self.MAX_NEW_TOKENS = 300
        self.N_CONTEXT_CHUNKS = 4
        
        # History
        self.MAX_HISTORY_MESSAGES = 5


# =============================================================================
# CONVERSATION HISTORY MANAGER
# =============================================================================

class ConversationHistory:
    """Manages conversation history with timestamps and metadata"""
    
    def __init__(self, max_messages: int = 5):
        self.history = []
        self.max_messages = max_messages
    
    def add(self, question: str, answer: str, sources: List[str] = None):
        """Add Q&A pair to history"""
        entry = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": sources or []
        }
        self.history.append(entry)
        
        # Keep only last N messages
        if len(self.history) > self.max_messages:
            self.history.pop(0)
    
    def get_history(self) -> List[Dict]:
        """Get full history"""
        return self.history
    
    def get_recent(self, n: int = 3) -> List[Dict]:
        """Get last N entries"""
        return self.history[-n:] if self.history else []
    
    def clear(self):
        """Clear history"""
        self.history.clear()
    
    def format_for_prompt(self) -> str:
        """Format history for inclusion in prompt"""
        if not self.history:
            return ""
        
        history_text = "\nConversation Memory (User Questions Only):\n"
        for i, entry in enumerate(self.history, 1):
            history_text += f"User Q{i}: {entry['question']}\n"
        return history_text + "\n"


# =============================================================================
# GUARD RAILS
# =============================================================================

class GuardRails:
    """Security and content filtering for user queries"""
    
    # Blocked topics
    BLOCKED_TOPICS = [
        "politics", "politician", "election", "government", "minister",
        "stock", "share", "investment", "crypto", "bitcoin",
        "weather", "forecast", "temperature", "rain",
        "movie", "film", "actor", "actress", "celebrity",
        "sports", "cricket", "football", "match",
        "code", "system prompt", "backend", "debug", "reveal",
        "instruction", "implementation", "algorithm",
        "artificial intelligence", "machine learning", "technology",
        "programming", "software", "computer", "science"
    ]
    
    # Security patterns
    SECURITY_PATTERNS = [
        r"show.*code", r"reveal.*prompt", r"system.*prompt",
        r"backend.*logic", r"debug.*mode", r"show.*instruction",
        r"implementation", r"share.*code", r"expose.*system"
    ]
    
    @staticmethod
    def check_query(question: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query is allowed.
        Returns: (is_allowed, error_message)
        """
        q_lower = question.lower().strip()
        
        # Check for security violations
        for pattern in GuardRails.SECURITY_PATTERNS:
            if re.search(pattern, q_lower):
                return False, (
                    "❌ **Security Guard Rail Activated**\n\n"
                    "I cannot reveal system internals, code, or implementation details.\n\n"
                    "Please ask questions about Hindu temples, Jyotirlingas, or Shakti Peethas."
                )
        
        # Check for blocked topics
        for topic in GuardRails.BLOCKED_TOPICS:
            if topic in q_lower:
                return False, (
                    f"⚠️ **Content Guard Rail Activated**\n\n"
                    f"I can only answer questions about Hindu temples.\n\n"
                    f"Topics like '{topic}' are outside my knowledge domain.\n\n"
                    f"Please ask about temples, Jyotirlingas, or Shakti Peethas."
                )
        
        return True, None


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

class TextProcessor:
    """Text cleaning and chunking utilities"""
    
    @staticmethod
    def robust_sentence_split(text: str) -> List[str]:
        """Split text into sentences with fallback mechanisms"""
        txt = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', txt)
        
        if len(sentences) < 3:
            sentences = re.split(r'(?<=[.!?;:])\s+', txt)
        
        if len(sentences) < 3:
            words = txt.split(" ")
            chunk_size = 50
            sentences = [" ".join(words[i:i+chunk_size]) 
                        for i in range(0, len(words), chunk_size)]
        
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = 800, 
                   overlap_sentences: int = 1) -> List[str]:
        """Chunk text with sentence overlap"""
        sentences = TextProcessor.robust_sentence_split(text)
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            if current and current_len + len(sent) + 1 > max_chars:
                chunks.append(" ".join(current))
                # Overlap
                if overlap_sentences > 0:
                    current = current[-overlap_sentences:]
                    current_len = len(" ".join(current))
                else:
                    current = []
                    current_len = 0
            
            current.append(sent)
            current_len += len(sent) + 1
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    @staticmethod
    def clean_generated_text(text: str) -> str:
        """Clean garbled/corrupted text from model output"""
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Remove weird special characters
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"•]+', ' ', text)
        # Remove repeated punctuation
        text = re.sub(r'([.!?,;:])\1+', r'\1', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences and stop at corruption
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_sentences = []
        for sentence in sentences:
            special_ratio = len(re.findall(r'[^\w\s]', sentence)) / max(len(sentence), 1)
            if special_ratio < 0.3 and len(sentence.split()) >= 3:
                clean_sentences.append(sentence)
            else:
                break
        return ' '.join(clean_sentences)
    
    @staticmethod
    def remove_duplicate_blocks(text: str) -> str:
        """Remove repeated lines/blocks in generated output"""
        lines = text.split("\n")
        seen = set()
        clean_lines = []
        for l in lines:
            l_clean = l.strip()
            if l_clean and l_clean not in seen:
                clean_lines.append(l_clean)
                seen.add(l_clean)
        return "\n".join(clean_lines)
    
    @staticmethod
    def format_as_bullets(text: str) -> str:
        """Ensure text is formatted as bullet points"""
        if not text:
            return text
        
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # If line doesn't start with bullet, add it
            if not line.startswith('•') and not line.startswith('-') and not line.startswith('*'):
                line = f"• {line}"
            # Standardize to •
            elif line.startswith('-') or line.startswith('*'):
                line = f"• {line[1:].strip()}"
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)


# =============================================================================
# MAIN RAG SYSTEM
# =============================================================================

class TempleRAG:
    """Main RAG system for temple information"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None
        self.gen_model = None
        self.tokenizer = None
        self.rag_pipe = None
        self.client = None
        self.collection = None
        self.conversation_history = ConversationHistory(config.MAX_HISTORY_MESSAGES)
        self.guard_rails = GuardRails()
        self.text_processor = TextProcessor()
    
    # -------------------------------------------------------------------------
    # DATABASE INITIALIZATION
    # -------------------------------------------------------------------------
    
    def initialize_database(self):
        """Initialize ChromaDB"""
        print("📦 Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name="temple_guider_rag_demo",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ ChromaDB ready.")
    
    def is_database_empty(self) -> bool:
        """Check if database needs to be populated"""
        return self.collection.count() == 0
    
    # -------------------------------------------------------------------------
    # PDF PROCESSING
    # -------------------------------------------------------------------------
    
    def extract_pdf_text(self, pdf_path: Path, max_pages: Optional[int] = None) -> List[Dict]:
        """Extract text from PDF"""
        reader = PdfReader(pdf_path)
        pages_text = []
        total_pages = len(reader.pages)
        
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)
        
        for i in range(total_pages):
            page = reader.pages[i]
            raw = page.extract_text() or ""
            text = raw.strip()
            pages_text.append({"page": i+1, "text": text})
        
        return pages_text
    
    def process_and_store_pdfs(self):
        """Process all PDFs and store in ChromaDB"""
        pdf_files = list(Path(self.config.PDF_DIR).glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.config.PDF_DIR}")
        
        print(f"\n📚 Processing {len(pdf_files)} PDFs...")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            print(f"\n📄 Processing: {pdf_file.name}")
            pages = self.extract_pdf_text(pdf_file, self.config.MAX_PAGES)
            
            for page in pages:
                page_num = page["page"]
                chunks = self.text_processor.chunk_text(
                    page["text"],
                    max_chars=self.config.CHUNK_MAX_CHARS,
                    overlap_sentences=self.config.CHUNK_OVERLAP_SENTENCES
                )
                
                for idx, chunk in enumerate(chunks):
                    doc_id = f"{pdf_file.stem}_p{page_num}_c{idx+1}"
                    all_documents.append({
                        "id": doc_id,
                        "pdf": pdf_file.name,
                        "page": page_num,
                        "text": chunk
                    })
        
        print(f"\n✅ Created {len(all_documents)} chunks")
        print("\n🔍 Creating embeddings...")
        
        # Create embeddings in batches
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i+batch_size]
            
            # Compute embeddings
            texts = [doc["text"] for doc in batch]
            embeddings = self.embed_model.encode(texts).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                ids=[doc["id"] for doc in batch],
                embeddings=embeddings,
                metadatas=[{"pdf": doc["pdf"], "page": doc["page"]} for doc in batch],
                documents=texts
            )
            
            print(f"   → Embedded {min(i+batch_size, len(all_documents))}/{len(all_documents)} chunks")
        
        print("\n✅ All embeddings stored in ChromaDB!")
    
    # -------------------------------------------------------------------------
    # MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    
    def initialize_models(self):
        """Initialize embedding and generation models"""
        print("🔍 Loading embedding model...")
        self.embed_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded.")
        
        print("\n🧠 Loading generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.GEN_MODEL)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            self.config.GEN_MODEL,
            torch_dtype="auto",
            device_map="cpu"
        )
        
        self.rag_pipe = pipeline(
            "text-generation",
            model=self.gen_model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.MAX_NEW_TOKENS,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.25,
        )
        print("✅ Generation model loaded.")
    
    # -------------------------------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------------------------------
    
    def retrieve_chunks(self, query: str, n_results: int = 4) -> Tuple[List[str], List[Dict], List[float]]:
        """Retrieve relevant chunks from ChromaDB"""
        query_embedding = self.embed_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        
        return docs, metas, dists
    
    # -------------------------------------------------------------------------
    # HARDCODED RESPONSES
    # -------------------------------------------------------------------------
    
    def get_jyotirlinga_list(self) -> str:
        """Return hardcoded list of 12 Jyotirlingas"""
        return """**The 12 Jyotirlingas in India:**

1. **Somnath** - Gujarat
2. **Mallikarjuna** - Srisailam, Andhra Pradesh
3. **Mahakaleswar** - Ujjain, Madhya Pradesh
4. **Omkareshwar** - Madhya Pradesh
5. **Kedarnath** - Uttarakhand
6. **Bhimashankar** - Pune, Maharashtra
7. **Vishwanath** - Varanasi, Uttar Pradesh
8. **Tryambakeshwar** - Nashik, Maharashtra
9. **Baidyanath** - Deogarh, Jharkhand
10. **Nageshwar** - Gujarat
11. **Rameshwar** - Rameshwaram, Tamil Nadu
12. **Grishneshwar** - Ellora, Aurangabad, Maharashtra

This is the list of jyothirlingas"""
    
    def get_shakti_peetha_list(self) -> str:
        """Return hardcoded list of 18 Maha Shakti Peethas"""
        return """**The 18 Maha Shakti Peethas in India:**

1. **Kamakhya** - Guwahati, Assam
2. **Kalighat** - Kolkata, West Bengal
3. **Ambaji** - Gujarat
4. **Jwalamukhi** - Himachal Pradesh
5. **Vaishno Devi** - Jammu & Kashmir
6. **Chamundeshwari** - Mysore, Karnataka
7. **Tulja Bhavani** - Maharashtra
8. **Vishalakshi** - Varanasi, Uttar Pradesh
9. **Naina Devi** - Himachal Pradesh
10. **Chhinnamasta** - Jharkhand
11. **Bhramari Devi** - West Bengal
12. **Dakshina Kalika** - West Bengal
13. **Kanaka Durga** - Vijayawada, Andhra Pradesh
14. **Tripura Sundari** - Tripura
15. **Mahalakshmi** - Kolhapur, Maharashtra
16. **Yogini** - Odisha
17. **Renuka** - Himachal Pradesh
18. **Shri Guhyeshwari** - Nepal

This is the list of Mahashaktipeetas      """
    
    # -------------------------------------------------------------------------
    # PROMPT BUILDING
    # -------------------------------------------------------------------------
    
    def build_rag_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Build RAG prompt with context and history"""
        # Select context chunks
        selected = []
        total_len = 0
        for chunk in context_chunks:
            if total_len + len(chunk) > self.config.MAX_CONTEXT_CHARS:
                break
            selected.append(chunk)
            total_len += len(chunk)
        
        context_text = "\n\n".join(selected)
        history_text = self.conversation_history.format_for_prompt()
        
        prompt = f"""You are an expert on Hindu temples in India.
Your ONLY knowledge source is the PDF chunks retrieved from the vector database.

CRITICAL FORMATTING RULE:
- You MUST answer in bullet points ONLY
- Start each point with a bullet (•)
- Use 5-7 bullet points maximum
- Keep each bullet concise and informative
- NO paragraphs, NO long explanations

ANSWER STRUCTURE (use bullet points for each):
• Temple Name & Location
• Special Features / Architecture
• Historical Period / Establishment Year
• Deity / Spiritual Significance
• Rituals / Practices (if in PDF)
• Legends / Cultural Notes (if in PDF)
• Additional factual details (if in PDF)

STRICT RULES:
1. Use ONLY facts from the provided PDF chunks
2. NEVER use outside knowledge
3. If information is missing, write "I don't know"
4. If temple not in PDF, say "I don't know about this temple"
5. Always end with: "Do not assume. Answered using only PDF-extracted information."

Retrieved Context:
{context_text}

{history_text}
Current Question:
{question}

Answer in bullet points:
"""
        
        return prompt.strip()
    
    # -------------------------------------------------------------------------
    # QUESTION ANSWERING
    # -------------------------------------------------------------------------
    
    def answer_question(self, question: str) -> str:
        """Main question answering function with guard rails"""
        
        # Clean question
        question_clean = question.replace("Answer:", "").strip()
        q_lower = question_clean.lower()
        
        # Check for Jyotirlinga list requests
        if any(keyword in q_lower for keyword in ['jyotirlinga', 'jyothirlinga', 'jythorilingas', 'list jyotirlinga', 'all jyotirlinga']):
            if 'list' in q_lower or 'all' in q_lower or 'name' in q_lower or '12' in q_lower:
                response = self.get_jyotirlinga_list()
                self.conversation_history.add(question_clean, response)
                return response
        
        # Check for Shakti Peetha list requests
        if any(keyword in q_lower for keyword in ['shakti', 'peetha', 'peethas', 'shakti peetha', 'maha shakti']):
            if 'list' in q_lower or 'all' in q_lower or 'name' in q_lower or '18' in q_lower or 'maha' in q_lower:
                response = self.get_shakti_peetha_list()
                self.conversation_history.add(question_clean, response)
                return response
        
        # Check for history queries
        if self._is_history_query(question_clean):
            return self._get_history_response(question_clean)
        
        # Check guard rails
        is_allowed, error_msg = self.guard_rails.check_query(question_clean)
        if not is_allowed:
            return error_msg
        
        # Retrieve context
        docs, metas, dists = self.retrieve_chunks(
            question_clean,
            n_results=self.config.N_CONTEXT_CHUNKS
        )
        
        if not docs:
            response = "• I don't know based on the provided documents.\n\nDo not assume. Answered using only PDF-extracted information."
            self.conversation_history.add(question_clean, response)
            return response
        
        # Build prompt
        prompt = self.build_rag_prompt(question_clean, docs)
        
        # Generate answer
        full_output = self.rag_pipe(
            prompt,
            max_new_tokens=self.config.MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.5,
            repetition_penalty=1.25,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]["generated_text"]
        
        # Extract answer
        answer = full_output[len(prompt):].strip()
        answer = self.text_processor.clean_generated_text(answer)
        answer = self.text_processor.remove_duplicate_blocks(answer)
        
        # Format as bullets
        answer = self.text_processor.format_as_bullets(answer)
        
        # Ensure disclaimer is present
        if "Do not assume" not in answer:
            answer += "\n\nDo not assume. Answered using only PDF-extracted information."
        
        # Store sources
        sources = [f"{m['pdf']} (p{m['page']})" for m in metas[:3]]
        
        # Add to history
        self.conversation_history.add(question_clean, answer, sources)
        
        return answer
    
    def _is_history_query(self, question: str) -> bool:
        """Check if query is about conversation history"""
        q_lower = question.lower().strip()
        history_keywords = [
            "last question", "previous question", "recent question",
            "last message", "previous message", "my last",
            "conversation history", "what did i ask", "show history"
        ]
        return any(keyword in q_lower for keyword in history_keywords)
    
    def _get_history_response(self, question: str) -> str:
        """Get formatted history response"""
        history = self.conversation_history.get_history()
        
        if not history:
            return "• No conversation history available yet."
        
        q_lower = question.lower()
        
        # Last question only
        if 'last' in q_lower and ('question' in q_lower or 'message' in q_lower):
            last_entry = history[-1]
            return f"📜 **Your last question was:**\n\n\"{last_entry['question']}\"\n\n*Asked at: {last_entry['timestamp']}*"
        
        # Full history
        response = "📜 **Your Conversation History:**\n\n"
        for i, entry in enumerate(history, 1):
            response += f"**Q{i}:** {entry['question']}\n"
            response += f"*({entry['timestamp']})*\n\n"
        
        return response


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize
    config = Config()
    rag = TempleRAG(config)
    
    # Setup database
    rag.initialize_database()
    
    if rag.is_database_empty():
        print("Processing PDFs for first time...")
        rag.initialize_models()
        rag.process_and_store_pdfs()
    else:
        print(f"Database ready with {rag.collection.count()} chunks")
        rag.initialize_models()
    
    # Test query
    answer = rag.answer_question("Tell me about Tirupati temple")
    print("\n📖 Answer:\n")
    print(answer)