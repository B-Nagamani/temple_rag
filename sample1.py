"""
Temple Guider AI - RAG System for Hindu Temples Information
Covers: Famous temples, 12 Jyotirlingas, 18 Maha Shakti Peethas
"""
'''
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional

# PDF processing
from pypdf import PdfReader

# Embeddings and models
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Vector DB
import chromadb

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration"""
    PDF_DIR = "/Users/nagamanibhukya/Downloads/rag sample/Templeslist"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    CHROMA_DB_DIR = "chroma_rag_demo"
    COLLECTION_NAME = "temple_guider_rag_demo"
    
    # Chunking parameters
    MAX_CHUNK_CHARS = 800
    OVERLAP_SENTENCES = 1
    MAX_PAGES_PER_PDF = None
    
    # Retrieval parameters
    TOP_K_CHUNKS = 5
    MAX_CONTEXT_CHARS = 2000
    
    # Generation parameters
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.3
    TOP_P = 0.85
    REPETITION_PENALTY = 1.3
    
    # History
    MAX_HISTORY_MESSAGES = 5


# =============================================================================
# PDF PROCESSING
# =============================================================================

class PDFProcessor:
    """Handle PDF extraction and text processing"""
    
    @staticmethod
    def extract_pdf_text(pdf_path: Path, max_pages: Optional[int] = None) -> List[Dict]:
        """Extract text from PDF pages"""
        reader = PdfReader(pdf_path)
        pages_text = []
        total_pages = len(reader.pages)
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        print(f"📄 Reading {pdf_path.name} ({total_pages} pages)...")
        
        for i in range(total_pages):
            page = reader.pages[i]
            text = (page.extract_text() or "").strip()
            pages_text.append({"page": i + 1, "text": text})
        
        return pages_text
    
    @staticmethod
    def robust_sentence_split(text: str) -> List[str]:
        """Split text into sentences with fallback mechanisms"""
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 3:
            sentences = re.split(r'(?<=[.!?;:])\s+', text)
        
        if len(sentences) < 3:
            words = text.split()
            chunk_size = 50
            sentences = [" ".join(words[i:i+chunk_size]) 
                        for i in range(0, len(words), chunk_size)]
        
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = 800, overlap_sentences: int = 1) -> List[str]:
        """Chunk text with sentence overlap"""
        sentences = PDFProcessor.robust_sentence_split(text)
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent) + 1
            
            if current and current_len + sent_len > max_chars:
                chunks.append(" ".join(current))
                
                if overlap_sentences > 0:
                    current = current[-overlap_sentences:]
                    current_len = len(" ".join(current))
                else:
                    current = []
                    current_len = 0
            
            current.append(sent)
            current_len += sent_len
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks


# =============================================================================
# RAG SYSTEM
# =============================================================================

class TempleRAG:
    """Main RAG system for Temple information"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None
        self.collection = None
        self.rag_pipe = None
        self.tokenizer = None
        self.conversation_history = []
    
    def initialize_models(self):
        """Initialize embedding and generation models ONLY"""
        print("🔍 Loading embedding model...")
        self.embed_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded.\n")
        
        print("🧠 Loading Qwen generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.GEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.GEN_MODEL,
            torch_dtype="auto",
            device_map="cpu"
        )
        
        self.rag_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.MAX_NEW_TOKENS,
            temperature=self.config.TEMPERATURE,
            top_p=self.config.TOP_P,
            repetition_penalty=self.config.REPETITION_PENALTY,
        )
        print("✅ Generation model loaded.\n")
    
    def initialize_database(self):
        """Initialize ChromaDB connection"""
        print("📦 Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)
        self.collection = client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Database connected. Current documents: {self.collection.count()}\n")
    
    def is_database_empty(self) -> bool:
        """Check if embeddings already exist"""
        return self.collection.count() == 0
    
    def process_and_store_pdfs(self):
        """Process PDFs and store embeddings (ONE-TIME OPERATION)"""
        pdf_files = list(Path(self.config.PDF_DIR).glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.config.PDF_DIR}")
        
        print(f"📚 Found {len(pdf_files)} PDF files\n")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            pages = PDFProcessor.extract_pdf_text(
                pdf_file, 
                max_pages=self.config.MAX_PAGES_PER_PDF
            )
            
            for page in pages:
                chunks = PDFProcessor.chunk_text(
                    page["text"],
                    max_chars=self.config.MAX_CHUNK_CHARS,
                    overlap_sentences=self.config.OVERLAP_SENTENCES
                )
                
                for idx, chunk in enumerate(chunks):
                    doc_id = f"{pdf_file.stem}_p{page['page']}_c{idx+1}"
                    all_documents.append({
                        "id": doc_id,
                        "pdf": pdf_file.name,
                        "page": page["page"],
                        "text": chunk
                    })
        
        print(f"✅ Created {len(all_documents)} chunks\n")
        
        print("📝 Creating and storing embeddings...")
        for idx, doc in enumerate(all_documents):
            embedding = self.embed_model.encode(doc["text"]).tolist()
            
            self.collection.add(
                ids=[doc["id"]],
                embeddings=[embedding],
                metadatas=[{"pdf": doc["pdf"], "page": doc["page"]}],
                documents=[doc["text"]]
            )
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(all_documents):
                print(f"   → {idx+1}/{len(all_documents)} embeddings stored")
        
        print("\n✅ All PDFs processed and embeddings stored!\n")
    
    def retrieve_chunks(self, query: str, n_results: int = None) -> Tuple[List, List, List]:
        """Retrieve relevant chunks from vector DB"""
        if n_results is None:
            n_results = self.config.TOP_K_CHUNKS
        
        query_embedding = self.embed_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return (
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    
    def is_list_question(self, question: str) -> bool:
        """Check if the question is asking for a list"""
        list_keywords = [
            'list', 'all 12', 'all 18', 'jyotirlinga', 'jyothirlinga', 'jythorilingas',
            'shakti peetha', 'names of', 'how many', 'which are', 'give me all',
            'tell me all', 'show me all', 'what are the 12', 'what are the 18'
        ]
        question_lower = question.lower()
        
        # Check if any keyword exists
        for keyword in list_keywords:
            if keyword in question_lower:
                return True
        
        return False
    
    def build_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Build prompt based on question type"""
        # Select context within char limit
        selected = []
        total_len = 0
        for chunk in context_chunks:
            if total_len + len(chunk) > self.config.MAX_CONTEXT_CHARS:
                break
            selected.append(chunk)
            total_len += len(chunk)
        
        context_text = "\n\n".join(selected)
        
        # Different prompts for different question types
        if self.is_list_question(question):
            prompt = f"""You are listing Hindu temples. Extract ONLY temple names and locations from the context.

CONTEXT:
{context_text}

QUESTION: {question}

STRICT INSTRUCTIONS:
- Extract ONLY temple names with their locations
- Format: TempleName in City, State
- Separate each with semicolon (;)
- No explanations, no descriptions, no extra text
- Just list: Temple1 in Location1; Temple2 in Location2; etc.

List:"""
        else:
            prompt = f"""You are a Hindu temple expert. Provide key facts from the context.

CONTEXT:
{context_text}

QUESTION: {question}

Provide 5-7 key facts. Write each as a complete sentence ending with period. No bullet points.

Answer:"""
        
        return prompt
    
    def format_list_response(self, text: str, question: str) -> str:
        """Format response as numbered list for list questions"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by semicolon first, then by period
        entries = re.split(r'[;.]', text)
        
        # Extract temple names and locations
        temples = []
        seen = set()
        
        for entry in entries:
            entry = entry.strip()
            if len(entry) < 5:
                continue
            
            # Pattern: "Temple in/at Location"
            match = re.search(r'([A-Za-z\s]+?)\s+(?:in|at)\s+([A-Za-z\s,]+)', entry, re.IGNORECASE)
            
            if match:
                temple_name = match.group(1).strip()
                location = match.group(2).strip()
            else:
                # Try splitting by "in"
                parts = entry.split(' in ', 1)
                if len(parts) == 2:
                    temple_name = parts[0].strip()
                    location = parts[1].strip()
                else:
                    continue
            
            # Clean temple name - remove common words
            temple_name = re.sub(r'\b(temple|jyotirlinga|jyothirlinga|peetha|shrine)\b', '', temple_name, flags=re.IGNORECASE).strip()
            
            # Skip if too short or duplicate
            if len(temple_name) < 3 or temple_name.lower() in seen:
                continue
            
            seen.add(temple_name.lower())
            temples.append((temple_name, location))
        
        # If not enough temples found, return default message
        if len(temples) < 3:
            # Fallback to hardcoded list for Jyotirlingas
            if 'jyotirlinga' in question.lower() or 'jyothirlinga' in question.lower() or 'jythorilingas' in question.lower():
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
12. **Grishneshwar** - Ellora, Aurangabad, Maharashtra"""
            
            # Fallback for Shakti Peethas
            elif 'shakti' in question.lower() or 'peetha' in question.lower():
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
18. **Shri Guhyeshwari** - Nepal"""
            
            else:
                return self.format_bullet_response(text)
        
        # Build formatted output
        if 'jyotirlinga' in question.lower() or 'jyothirlinga' in question.lower() or 'jythorilingas' in question.lower():
            output = "**The 12 Jyotirlingas in India:**\n\n"
        elif 'shakti' in question.lower():
            output = "**The 18 Maha Shakti Peethas in India:**\n\n"
        else:
            output = "**List of Hindu Temples:**\n\n"
        
        for i, (temple, location) in enumerate(temples[:15], 1):
            output += f"{i}. **{temple}** - {location}\n"
        
        return output
    
    def format_bullet_response(self, text: str) -> str:
        """Format response as bullet points for specific temple questions"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove all existing bullets
        text = re.sub(r'[•\-\*]\s*', '', text)
        
        # Split by sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        # Create clean bullets
        bullets = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip if too short
            if len(sentence) < 20:
                continue
            
            # Remove unwanted phrases
            unwanted = [
                'to provide more detailed information',
                'please note that',
                'additional notes',
                'according to',
                'information not available',
                'as mentioned',
                'stated above'
            ]
            
            skip = False
            for phrase in unwanted:
                if phrase in sentence.lower():
                    skip = True
                    break
            
            if skip:
                continue
            
            # Clean sentence
            sentence = re.sub(r'\([^)]*\)', '', sentence)
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            
            # Check duplicates
            if sentence.lower() in seen:
                continue
            seen.add(sentence.lower())
            
            bullets.append(f"• {sentence}")
            
            # Limit to 7 bullets
            if len(bullets) >= 7:
                break
        
        return "\n\n".join(bullets) if bullets else "• No specific information found in the documents"
    
    def answer_question(self, question: str) -> str:
        """Full RAG pipeline"""
        question = question.strip()
        
        # Check for history queries
        if any(phrase in question.lower() for phrase in 
               ["conversation history", "what did i ask", "previous question"]):
            if self.conversation_history:
                response = "**Your conversation history:**\n\n"
                for i, msg in enumerate(self.conversation_history, 1):
                    response += f"{i}. Q: {msg['question']}\n"
                return response
            return "No conversation history yet."
        
        # HARDCODED RESPONSES FOR LIST QUESTIONS
        question_lower = question.lower()
        
        # Check for Jyotirlinga list questions
        if any(keyword in question_lower for keyword in ['jyotirlinga', 'jyothirlinga', 'jythorilingas']):
            if any(word in question_lower for word in ['list', 'all', 'names', 'which', 'what are', 'give', 'show']):
                answer = """**The 12 Jyotirlingas in India:**

1. **Somnath** - Veraval, Gujarat
2. **Mallikarjuna** - Srisailam, Andhra Pradesh
3. **Mahakaleshwar** - Ujjain, Madhya Pradesh
4. **Omkareshwar** - Khandwa, Madhya Pradesh
5. **Kedarnath** - Rudraprayag, Uttarakhand
6. **Bhimashankar** - Pune, Maharashtra
7. **Vishwanath** - Varanasi, Uttar Pradesh
8. **Tryambakeshwar** - Nashik, Maharashtra
9. **Vaidyanath** - Deoghar, Jharkhand
10. **Nageshwar** - Dwarka, Gujarat
11. **Rameshwar** - Rameswaram, Tamil Nadu
12. **Grishneshwar** - Ellora, Aurangabad, Maharashtra"""
                
                # Store in history
                self.conversation_history.append({
                    "question": question,
                    "answer": answer
                })
                
                if len(self.conversation_history) > self.config.MAX_HISTORY_MESSAGES:
                    self.conversation_history.pop(0)
                
                return answer
        
        # Check for Shakti Peetha list questions
        if any(keyword in question_lower for keyword in ['shakti', 'peetha', 'peeth', 'mahashakti', 'shaktha', 'shakthi', 'mahashaktha', 'mahashakthi']):
            if any(word in question_lower for word in ['list', 'all', 'names', 'which', 'what are', 'give', 'show', '18', 'the']):
                answer = """**The 18 Maha Shakti Peethas in India:**

1. **Kamakhya Temple** - Guwahati, Assam
2. **Kalighat Temple** - Kolkata, West Bengal
3. **Tara Tarini Temple** - Berhampur, Odisha
4. **Dakshineswar Kali Temple** - Kolkata, West Bengal
5. **Jwalamukhi Temple** - Kangra, Himachal Pradesh
6. **Vaishno Devi Temple** - Katra, Jammu & Kashmir
7. **Chamundeshwari Temple** - Mysore, Karnataka
8. **Tulja Bhavani Temple** - Tuljapur, Maharashtra
9. **Ambaji Temple** - Banaskantha, Gujarat
10. **Vishalakshi Temple** - Varanasi, Uttar Pradesh
11. **Chhinnamasta Temple** - Rajrappa, Jharkhand
12. **Naina Devi Temple** - Bilaspur, Himachal Pradesh
13. **Bhramari Devi Temple** - Jalpaiguri, West Bengal
14. **Kanaka Durga Temple** - Vijayawada, Andhra Pradesh
15. **Tripura Sundari Temple** - Udaipur, Tripura
16. **Mahalakshmi Temple** - Kolhapur, Maharashtra
17. **Renuka Devi Temple** - Sirmaur, Himachal Pradesh
18. **Attukkal Bhagavathy Temple** - Thiruvananthapuram, Kerala"""
                
                # Store in history
                self.conversation_history.append({
                    "question": question,
                    "answer": answer
                })
                
                if len(self.conversation_history) > self.config.MAX_HISTORY_MESSAGES:
                    self.conversation_history.pop(0)
                
                return answer
        
        # FOR ALL OTHER QUESTIONS - Use RAG
        # Retrieve context
        docs, metas, dists = self.retrieve_chunks(question)
        
        if not docs:
            return "No relevant information found in the temple documents."
        
        # Build prompt
        prompt = self.build_prompt(question, docs)
        
        # Generate answer
        full_output = self.rag_pipe(
            prompt,
            max_new_tokens=self.config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=self.config.TEMPERATURE,
            top_p=self.config.TOP_P,
            repetition_penalty=self.config.REPETITION_PENALTY,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]["generated_text"]
        
        # Extract answer
        answer = full_output[len(prompt):].strip()
        
        # Format as bullet points
        answer = self.format_bullet_response(answer)
        
        # Store in history
        self.conversation_history.append({
            "question": question,
            "answer": answer
        })
        
        if len(self.conversation_history) > self.config.MAX_HISTORY_MESSAGES:
            self.conversation_history.pop(0)
        
        return answer
