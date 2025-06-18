import streamlit as st
import torch
import tempfile
import os
from dotenv import load_dotenv
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bkai-foundation-models/vietnamese-bi-encoder")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "lmsys/vicuna-7b-v1.5")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# Streamlit page configuration
st.set_page_config(
    page_title="RAG - Hỏi đáp tài liệu AIO",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

@st.cache_resource
def load_embeddings():
    """Load and cache the embedding model"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Lỗi khi tải embedding model: {str(e)}")
        return None

@st.cache_resource
def load_llm():
    """Load and cache the LLM model"""
    try:
        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto",
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Integrate with LangChain
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm
    except Exception as e:
        st.error(f"Lỗi khi tải LLM model: {str(e)}")
        return None

def format_docs(docs):
    """Format retrieved documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def process_pdf(uploaded_file, embeddings, llm):
    """Process uploaded PDF and create RAG chain"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Initialize semantic chunker
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=int(os.getenv("SEMANTIC_BUFFER_SIZE", "1")),
            breakpoint_threshold_type=os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile"),
            breakpoint_threshold_amount=int(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT", "95")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "500")),
            add_start_index=True
        )
        
        # Split documents into chunks
        docs = semantic_splitter.split_documents(documents)
        
        # Create vector database
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Get RAG prompt
        prompt = hub.pull("rlm/rag-prompt")
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return rag_chain, len(docs)
        
    except Exception as e:
        # Clean up temporary file in case of error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise e

# Main UI
st.title("📚 RAG - Hỏi đáp tài liệu bài học AIO")
st.markdown("""
### Hướng dẫn sử dụng:
1. **Tải models**: Ứng dụng sẽ tự động tải các models cần thiết khi khởi chạy lần đầu
2. **Upload PDF**: Chọn file PDF tài liệu bài học cần hỏi đáp
3. **Xử lý PDF**: Nhấn nút "Xử lý PDF" để phân tích và tạo vector database
4. **Đặt câu hỏi**: Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời

---
""")

# Load models section
if not st.session_state.models_loaded:
    st.info("🔄 Đang tải models... Vui lòng đợi trong giây lát.")
    
    # Load embedding model
    with st.spinner("Đang tải Embedding Model..."):
        st.session_state.embeddings = load_embeddings()
    
    if st.session_state.embeddings is not None:
        st.success("✅ Embedding Model đã được tải thành công!")
        
        # Load LLM
        with st.spinner("Đang tải Large Language Model (Vicuna 7B)... Đây là bước tốn thời gian nhất."):
            st.session_state.llm = load_llm()
        
        if st.session_state.llm is not None:
            st.success("✅ Large Language Model đã được tải thành công!")
            st.session_state.models_loaded = True
            st.rerun()
        else:
            st.error("❌ Không thể tải Large Language Model. Vui lòng thử lại.")
            st.stop()
    else:
        st.error("❌ Không thể tải Embedding Model. Vui lòng thử lại.")
        st.stop()
else:
    st.success("✅ Tất cả models đã được tải thành công!")
    
    # File upload section
    st.subheader("📄 Upload và xử lý PDF")
    uploaded_file = st.file_uploader(
        "Chọn file PDF tài liệu bài học:",
        type="pdf",
        help="Chọn file PDF chứa nội dung bài học cần hỏi đáp"
    )
    
    if uploaded_file is not None:
        st.info(f"📁 Đã chọn file: {uploaded_file.name}")
        
        if st.button("🔄 Xử lý PDF", type="primary"):
            with st.spinner("Đang xử lý PDF và tạo vector database..."):
                try:
                    rag_chain, num_chunks = process_pdf(
                        uploaded_file, 
                        st.session_state.embeddings, 
                        st.session_state.llm
                    )
                    st.session_state.rag_chain = rag_chain
                    st.session_state.pdf_processed = True
                    st.success(f"✅ Xử lý PDF thành công! Đã tạo {num_chunks} chunks.")
                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý PDF: {str(e)}")
    
    # Q&A section
    if st.session_state.pdf_processed and st.session_state.rag_chain is not None:
        st.subheader("💬 Hỏi đáp với tài liệu")
        
        question = st.text_input(
            "Nhập câu hỏi của bạn:",
            placeholder="Ví dụ: Nội dung chính của bài học này là gì?",
            help="Đặt câu hỏi liên quan đến nội dung trong file PDF đã upload"
        )
        
        if question:
            with st.spinner("Đang trả lời..."):
                try:
                    response = st.session_state.rag_chain.invoke(question)
                    
                    # Clean up response (remove "Answer:" prefix if present)
                    if response.startswith("Answer:"):
                        response = response[7:].strip()
                    
                    st.subheader("📝 Câu trả lời:")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi trả lời câu hỏi: {str(e)}")
    
    elif not st.session_state.pdf_processed:
        st.info("📋 Vui lòng upload và xử lý file PDF trước khi đặt câu hỏi.")

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ Thông tin dự án")
    st.markdown("""
    **Công nghệ sử dụng:**
    - 🤖 LangChain Framework
    - 🧠 Vicuna 7B LLM
    - 🔍 Vietnamese Bi-Encoder
    - 📊 ChromaDB Vector Store
    - 🎨 Streamlit Interface
    
    **Tính năng:**
    - ✂️ Semantic Chunking
    - 🔍 Vector Similarity Search
    - 💾 Model Caching
    - 🚀 4-bit Quantization
    """)
    
    st.header("⚙️ Cấu hình hiện tại")
    st.code(f"""
    Embedding Model: {EMBEDDING_MODEL_NAME}
    LLM Model: {LLM_MODEL_NAME}
    Max New Tokens: {MAX_NEW_TOKENS}
    Models Loaded: {st.session_state.models_loaded}
    PDF Processed: {st.session_state.pdf_processed}
    """)