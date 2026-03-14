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
        self.MAX_CONTEXT_CHARS = 2000
        self.MAX_NEW_TOKENS = 500
        self.N_CONTEXT_CHUNKS = 5
        
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
    
    def get_recent(self, n: int = 5) -> List[Dict]:
        """Get last N entries"""
        return self.history[-n:] if self.history else []
    
    def clear(self):
        """Clear history"""
        self.history.clear()
    
    def format_for_prompt(self) -> str:
        """Format history for inclusion in prompt"""
        if not self.history:
            return ""
        
        history_text = "\nRecent Conversation:\n"
        recent = self.get_recent(3)
        for i, entry in enumerate(recent, 1):
            history_text += f"Q{i}: {entry['question']}\n"
        return history_text + "\n"


# =============================================================================
# GUARD RAILS (FIXED AND ENFORCED)
# =============================================================================

class GuardRails:
    """Security and content filtering for user queries"""
    
    # Comprehensive security patterns - MUST be blocked
    SECURITY_PATTERNS = [
        # System/Prompt queries
        r"\b(show|reveal|tell|display|explain|share|give|what|whats|what's)\b.*(prompt|instruction|system)",
        r"\b(your|the|my|this)\b.*(prompt|instruction|system|code)",
        r"\bprompt\b",
        r"\binstruction[s]?\b",
        
        # Code/Debug queries
        r"\b(debug|show|reveal|share)\b.*(code|implementation|logic)",
        r"\bdebug\s+(my|your|the|this|code)",
        r"\bshow\s+(code|implementation)",
        
        # Backend queries
        r"\b(backend|server|api|database)\b.*(logic|code|work)",
        r"\bhow\s+(do|does|did|is)\s+(this|you|it|system)\s+(work|built|coded|implemented)",
        
        # Direct attacks
        r"\bignore\s+(previous|above|all)",
        r"\bbypass\b",
        r"\bhack\b",
        r"\bexploit\b",
        r"\boverride\b",
        
        # Meta queries about the system
        r"\bhow\s+are\s+you\s+(made|built|created|programmed)",
        r"\bwhat\s+(model|ai|system)\s+are\s+you",
        r"\byour\s+(architecture|design|structure)",
    ]
    
    # Out of domain patterns
    OUT_OF_DOMAIN_PATTERNS = [
        r"\b(actor|actress|movie|film|cinema|bollywood)\b",
        r"\b(cricket|football|sports|player)\b",
        r"\b(politician|minister|president|prime minister)\b",
        r"\b(recipe|cooking|food|dish)\b",
        r"\b(weather|temperature|climate)\b",
    ]
    
    # Temple-related keywords for validation
    TEMPLE_KEYWORDS = [
        'temple', 'mandir', 'jyotirlinga', 'jyothirlinga', 'shakti', 'peetha', 
        'shrine', 'deity', 'god', 'goddess', 'worship', 'pilgrimage', 'darshan',
        'prasad', 'puja', 'aarti', 'bhakti', 'devotee', 'sacred', 'holy',
        'lord', 'shiva', 'vishnu', 'durga', 'ganesh', 'hanuman', 'ram',
        'krishna', 'lakshmi', 'saraswati', 'kali', 'parvati',
        'tirupati', 'somnath', 'kedarnath', 'varanasi', 'kashi', 'rameshwaram',
        'dwarka', 'badrinath', 'puri', 'jagannath', 'srisailam', 'ujjain',
        'ttd', 'tirumala', 'venkateswara', 'balaji'
    ]
    
    @staticmethod
    def check_query(question: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query is allowed.
        Returns: (is_allowed, error_message)
        """
        q_lower = question.lower().strip()
        
        # STEP 1: Check for security violations - BLOCK IMMEDIATELY
        for pattern in GuardRails.SECURITY_PATTERNS:
            if re.search(pattern, q_lower, re.IGNORECASE):
                print(f"🚫 BLOCKED: Security pattern matched - {pattern}")
                return False, "security_violation"
        
        # STEP 2: Check if it's a greeting (allow these)
        greetings = ['hi', 'hello', 'hey', 'namaste', 'hii', 'helo', 'hai']
        if q_lower in greetings or (len(q_lower) < 15 and any(g in q_lower for g in greetings)):
            return True, None
        
        # STEP 3: Check if it's a history query (allow these)
        history_keywords = ['last question', 'previous question', 'recent question',
                           'last message', 'my last', 'conversation history', 
                           'what did i ask', 'show history', 'chat history']
        if any(kw in q_lower for kw in history_keywords):
            return True, None
        
        # STEP 4: Check if it contains temple-related keywords
        has_temple_keyword = any(kw in q_lower for kw in GuardRails.TEMPLE_KEYWORDS)
        
        # STEP 5: Check for out-of-domain topics
        for pattern in GuardRails.OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, q_lower, re.IGNORECASE):
                # If out-of-domain AND no temple keyword, block it
                if not has_temple_keyword:
                    print(f"🚫 BLOCKED: Out of domain - {pattern}")
                    return False, "out_of_domain"
        
        # STEP 6: For questions without temple keywords, be cautious
        question_words = ['what', 'who', 'where', 'when', 'how', 'why', 'tell', 'list', 'show']
        has_question_word = any(qw in q_lower for qw in question_words)
        
        if has_question_word and not has_temple_keyword:
            # It's a question but not about temples
            print(f"🚫 BLOCKED: Question without temple context")
            return False, "not_temple_related"
        
        # All checks passed
        return True, None
    
    @staticmethod
    def get_blocked_response(block_reason: str) -> List[str]:
        """Get appropriate response for blocked queries"""
        
        responses = {
            "security_violation": [
                "I can only provide information about Hindu temples",
                "Ask me about: Jyotirlingas, Shakti Peethas, or specific temples",
                "Try: 'Tell me about Tirupati temple' or 'List all Jyotirlingas'"
            ],
            "out_of_domain": [
                "I specialize only in Hindu temple information",
                "I cannot answer questions about movies, actors, sports, or celebrities",
                "Ask me about: Temple history, architecture, locations, or significance"
            ],
            "not_temple_related": [
                "Please ask questions related to Hindu temples only",
                "I can help with: Temple locations, history, deities, and pilgrimage info",
                "Example: 'Where is Kedarnath temple?' or 'Tell me about Somnath'"
            ]
        }
        
        return responses.get(block_reason, responses["security_violation"])
    
    # Meta-commentary patterns to filter from responses
    META_PATTERNS = [
        r"based on the (pdf|information|context|document)",
        r"according to the (pdf|information|context|document)",
        r"from the (pdf|information|context|document)",
        r"the (pdf|information|context|document) (states|mentions|says)",
        r"answered using",
        r"information extracted",
        r"pdf-extracted",
        r"do not assume",
        r"(here is|here are) (the )?(information|details|facts)",
        r"let me (provide|give|share)",
        r"i found",
        r"according to my knowledge",
        r"based on my understanding"
    ]
    
    # Unwanted words/phrases
    UNWANTED_PHRASES = [
        "strategies", "strats", "strategy", "strategic",
        "summarize", "summary", "in summary", "to summarize",
        "in conclusion", "to conclude", "overall"
    ]
    
    @staticmethod
    def clean_response(text: str) -> str:
        """Remove meta-commentary and unwanted phrases from response"""
        cleaned = text
        
        # Remove meta-commentary patterns
        for pattern in GuardRails.META_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Remove unwanted phrases
        for phrase in GuardRails.UNWANTED_PHRASES:
            cleaned = re.sub(r'\b' + re.escape(phrase) + r'\b', "", cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned


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
        # Remove non-ASCII characters except bullet
        text = re.sub(r'[^\x00-\x7F•]+', ' ', text)
        # Clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_bullet_points(text: str, max_bullets: int = 7) -> List[str]:
        """Extract and format bullet points, limit to max_bullets"""
        # Split by newlines and bullet patterns
        lines = re.split(r'\n|(?=•)', text)
        bullet_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip meta-commentary and disclaimers
            if any(skip in line.lower() for skip in [
                'do not assume', 'pdf-extracted', 'answered using', 
                'based on', 'according to', 'information extracted',
                'from the pdf', 'the document states', 'summarize',
                'in summary', 'strategy', 'strats', 'strategic'
            ]):
                continue
            
            # Clean up bullet formatting
            if line.startswith('•'):
                line = line[1:].strip()
            elif line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()
            
            # Add if it's meaningful content
            if len(line) > 15 and not line.startswith('#'):
                # Apply guard rails to clean the line
                line = GuardRails.clean_response(line)
                if line and len(line) > 15:  # Re-check after cleaning
                    bullet_lines.append(line)
        
        # Limit to max_bullets
        bullet_lines = bullet_lines[:max_bullets]
        
        return bullet_lines if bullet_lines else ["Information not available in knowledge base"]


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
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        print("✅ Generation model loaded.")
    
    # -------------------------------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------------------------------
    
    def retrieve_chunks(self, query: str, n_results: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
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
    
    def get_jyotirlinga_list(self) -> List[str]:
        """Return list of 12 Jyotirlingas as bullet points"""
        return [
            "Somnath - Veraval, Gujarat",
            "Mallikarjuna - Srisailam, Andhra Pradesh",
            "Mahakaleswar - Ujjain, Madhya Pradesh",
            "Omkareshwar - Khandwa, Madhya Pradesh",
            "Kedarnath - Rudraprayag, Uttarakhand",
            "Bhimashankar - Pune, Maharashtra",
            "Vishwanath (Kashi) - Varanasi, Uttar Pradesh",
            "Tryambakeshwar - Nashik, Maharashtra",
            "Baidyanath - Deoghar, Jharkhand",
            "Nageshwar - Dwarka, Gujarat",
            "Rameshwar - Rameshwaram, Tamil Nadu",
            "Grishneshwar - Ellora, Aurangabad, Maharashtra"
        ]
    
    def get_shakti_peetha_list(self) -> List[str]:
        """Return list of 18 Maha Shakti Peethas as bullet points"""
        return [
            "Kamakhya - Guwahati, Assam",
            "Kalighat - Kolkata, West Bengal",
            "Ambaji - Banaskantha, Gujarat",
            "Jwalamukhi - Kangra, Himachal Pradesh",
            "Vaishno Devi - Katra, Jammu & Kashmir",
            "Chamundeshwari - Mysore, Karnataka",
            "Tulja Bhavani - Tuljapur, Maharashtra",
            "Vishalakshi - Varanasi, Uttar Pradesh",
            "Naina Devi - Bilaspur, Himachal Pradesh",
            "Chhinnamasta - Rajrappa, Jharkhand",
            "Bhramari Devi - Jalpaiguri, West Bengal",
            "Dakshina Kalika - Kolkata, West Bengal",
            "Kanaka Durga - Vijayawada, Andhra Pradesh",
            "Tripura Sundari - Udaipur, Tripura",
            "Mahalakshmi - Kolhapur, Maharashtra",
            "Yogini - Hirapur, Odisha",
            "Renuka - Sirmaur, Himachal Pradesh",
            "Guhyeshwari - Kathmandu, Nepal"
        ]
    
    def get_greeting_response(self) -> List[str]:
        """Return friendly greeting as bullet points"""
        return [
            "🙏 Namaste! Welcome to Temple Guider AI",
            "I can help you with information about Hindu temples in India",
            "Ask me about Jyotirlingas, Shakti Peethas, or any specific temple",
            "Try: 'Tell me about Tirupati temple' or 'List all Jyotirlingas'"
        ]
    
    # -------------------------------------------------------------------------
    # PROMPT BUILDING
    # -------------------------------------------------------------------------
    
    def build_rag_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Build RAG prompt with context - NO SUMMARIZE"""
        # Select context chunks
        selected = []
        total_len = 0
        for chunk in context_chunks:
            if total_len + len(chunk) > self.config.MAX_CONTEXT_CHARS:
                break
            selected.append(chunk)
            total_len += len(chunk)
        
        context_text = "\n\n".join(selected)
        
        prompt = f"""You are a Hindu temple information assistant. Provide factual information ONLY.

Temple Information:
{context_text}

Question: {question}

Provide 5-7 concise factual points. Each point on NEW line starting with •
DO NOT include strategies, summaries, or meta-commentary.
ONLY provide direct facts about the temple.

•"""
        
        return prompt.strip()
    
    # -------------------------------------------------------------------------
    # QUESTION ANSWERING
    # -------------------------------------------------------------------------
    
    def answer_question(self, question: str) -> List[str]:
        """Main question answering function - returns list of bullet points"""
        
        # Clean question
        question_clean = question.replace("Answer:", "").strip()
        q_lower = question_clean.lower()
        
        # Handle greetings (before guard rails)
        if any(greeting in q_lower for greeting in ['hi', 'hello', 'hey', 'namaste', 'hii', 'helo']):
            if len(q_lower) < 15:  # Simple greeting
                response = self.get_greeting_response()
                self.conversation_history.add(question_clean, "\n".join(response))
                return response
        
        # Check for Jyotirlinga list requests (before guard rails)
        if any(keyword in q_lower for keyword in ['jyotirlinga', 'jyothirlinga', 'jythorilingas']):
            if any(trigger in q_lower for trigger in ['list', 'all', 'name', '12', 'which are']):
                response = self.get_jyotirlinga_list()
                self.conversation_history.add(question_clean, "\n".join(response))
                return response
        
        # Check for Shakti Peetha list requests (before guard rails)
        if any(keyword in q_lower for keyword in ['shakti', 'peetha', 'peethas', 'shakti peetha']):
            if any(trigger in q_lower for trigger in ['list', 'all', 'name', '18', 'maha', 'which are']):
                response = self.get_shakti_peetha_list()
                self.conversation_history.add(question_clean, "\n".join(response))
                return response
        
        # Check for history queries (before guard rails)
        if self._is_history_query(question_clean):
            return self._get_history_response(question_clean)
        
        # ⚡ CRITICAL: CHECK GUARD RAILS - ENFORCE BLOCKING
        is_allowed, block_reason = self.guard_rails.check_query(question_clean)
        if not is_allowed:
            print(f"🚫 GUARD RAIL ACTIVATED: {block_reason}")
            response = self.guard_rails.get_blocked_response(block_reason)
            self.conversation_history.add(question_clean, "\n".join(response))
            return response
        
        # Retrieve context
        docs, metas, dists = self.retrieve_chunks(
            question_clean,
            n_results=self.config.N_CONTEXT_CHUNKS
        )
        
        if not docs:
            response = [
                "Information not available in my knowledge base",
                "Please try asking about other temples",
                "Try: 'List all Jyotirlingas' or 'Tell me about Tirupati temple'"
            ]
            self.conversation_history.add(question_clean, "\n".join(response))
            return response
        
        # Build prompt
        prompt = self.build_rag_prompt(question_clean, docs)
        
        # Generate answer
        try:
            full_output = self.rag_pipe(
                prompt,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]["generated_text"]
            
            # Extract answer after prompt
            answer = full_output[len(prompt):].strip()
            
            # Add back the • since prompt ends with it
            if not answer.startswith('•'):
                answer = '• ' + answer
            
            answer = self.text_processor.clean_generated_text(answer)
            
            # Extract bullet points with guard rails
            bullet_list = self.text_processor.extract_bullet_points(answer, max_bullets=7)
            
            # Ensure we have good content
            if len(bullet_list) < 3 or all(len(b) < 20 for b in bullet_list):
                bullet_list = [
                    "Information extracted from knowledge base",
                    "Please try rephrasing your question for better results",
                    "Or ask: 'List all Jyotirlingas'"
                ]
            
        except Exception as e:
            bullet_list = [
                "Unable to generate response",
                "Please try rephrasing your question"
            ]
        
        # Store sources
        sources = [f"{m['pdf']} (p{m['page']})" for m in metas[:3]]
        
        # Add to history
        self.conversation_history.add(question_clean, "\n".join(bullet_list), sources)
        
        return bullet_list
    
    def _is_history_query(self, question: str) -> bool:
        """Check if query is about conversation history"""
        q_lower = question.lower().strip()
        
        # Check for history keywords
        history_keywords = [
            "last question", "previous question", "recent question",
            "last message", "previous message", "my last",
            "conversation history", "what did i ask", "show history",
            "my messages", "chat history", "last 5", "5 last",
            "my 5", "show my", "what was my"
        ]
        
        return any(keyword in q_lower for keyword in history_keywords)
    
    def _get_history_response(self, question: str) -> List[str]:
        """Get formatted history response - SHOW LAST 5 WITHOUT TIMESTAMP"""
        history = self.conversation_history.get_recent(5)
        
        if not history:
            return [
                "No conversation history available yet",
                "Start asking about temples!"
            ]
        
        # Always show up to 5 messages without timestamp
        response = []
        for i, entry in enumerate(history, 1):
            response.append(f"Q{i}: {entry['question']}")
        
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
    for bullet in answer:
        print(f"• {bullet}")
