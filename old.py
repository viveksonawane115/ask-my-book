#Load env veriable
import time
start_time=time.perf_counter()
from dotenv import load_dotenv
load_dotenv()
import os 

#Load text document
from langchain_community.document_loaders import TextLoader

#Split the document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Convert chunks into embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

#The vector database
from langchain_chroma import Chroma

#LLM Model
from langchain_openai import ChatOpenAI

#Prompt Builder
from langchain_core.prompts import ChatPromptTemplate

#Initialise LLM here
llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18",temperature=0)

from langchain.retrievers.multi_query import MultiQueryRetriever 

###Block-2##
#Step - 1 : Load the document 
loader=TextLoader("book.txt",encoding="utf-8-sig")
documents=loader.load()
print(f"Document Loaded!")
print(f"Total characters in document:{len(documents[0].page_content)}")

#Step - 2 : Split the document into chunk
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
    add_start_index=True
)

chunks =splitter.split_documents(documents)
print(f"Splitting done !")
print(f"Total chunks created: {len(chunks)}")
print(f"\n Metadata : \n {chunks[0].metadata}")
print(f"Content:\n{chunks[0].page_content}")
print(f"\nMetadata:{chunks[0].metadata}")
print(f"Chunk character length : {len(chunks[0].page_content)}")


end_time=time.perf_counter()
totaltime=end_time-start_time
print(f"Total time taken for execution is {totaltime}")


#Step 3 - Create Embedding Model

embeddings_openai=OpenAIEmbeddings()
embeddings_hf=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

openai_vector_store=Chroma(
    collection_name="openai_collection",
    embedding_function=embeddings_openai,
    persist_directory="./openai_chroma_db"
)

if openai_vector_store._collection.count()==0:
    openai_vector_store.add_documents(chunks)
else:
    print(f"OpenAI collection already has {openai_vector_store._collection.count()} chunks — skipping embedding!")
    pass


hf_vector_store=Chroma(
    collection_name="hf_collection",
    embedding_function=embeddings_hf,
    persist_directory="./hf_chroma_db"
)

if hf_vector_store._collection.count()==0:
    hf_vector_store.add_documents(chunks)
else:
    print(f"HuggingFace collection already has {hf_vector_store._collection.count()} chunks — skipping embedding!")
    pass


#Step 4 - Ask Question
while True:
    
    
    question=input("\n You Question :")
    print(f"\n Q : {question}")

    if question.strip().lower() == "exit":
        print("\nExiting...")
        break

    #Step 5 - Retrieve and Compare

    #Retrieve from openai
    openai_results=openai_vector_store.similarity_search_with_score(query=question,k=3)

    #Retrive from Hugging Face
    hf_results=hf_vector_store.similarity_search_with_score(query=question,k=3)

    
    
    
    # Compare common chunks exxtracted by hugging face and openai
    openai_positions=set([doc.metadata.get('start_index') for doc,score in openai_results])
    hf_positions=set([doc.metadata.get('start_index') for doc ,score in hf_results])

    common=openai_positions.intersection(hf_positions)

    print(f"\n Common Chunks:{common}")

    only_openai=openai_positions - hf_positions
    print(f"only_openai chunks:{only_openai}")

    only_hf=hf_positions - openai_positions
    print(f"only_hf chunks:{only_hf}")

    print(f"Chunks found by BOTH models:{len(common)}")
    print(f"Chunks found ONLY by OpenAI:{len(only_openai)}")
    print(f"Chunks found ONLY by HuggingFace:{len(only_hf)}")

    if len(common) == 3:
        print("\n Both models agreed completely")
    elif len(common)==0:
        print("\n Models Disagreed completely")
    else:
        print(f"Models aggreed partially common chunks: {len(common)} different chunks: {len(only_openai)+len(only_hf)} ")



    #Step 6 - Generate and display answer

    prompt=ChatPromptTemplate.from_messages([("system","""You are an expert assitant. Answer ONLY from the context rovided below. If the answer is not the context,say 'Not found in the book.Be clear , concise and specific"""),("human","Context:\n{context}\n\n question:{question}")])

    def get_answers(question,results):
        context="\n\n".join([doc.page_content for doc,score in results])
        chain=prompt|llm
        response = chain.invoke({"context":context,"question":question})
        return response.content

    openai_answer=get_answers(question,openai_results)
    print(f"\n openai_answer = ",openai_answer)
    hf_answer=get_answers(question,hf_results)
    print(f"\n hf_answer = ",hf_answer)

