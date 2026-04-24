import json
import os
import chromadb
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
# Llama 3 8B is extremely fast and handles JSON formatting perfectly
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def init_db():
    """Initializes a local ChromaDB and loads candidates as vectors."""
    # This creates a local folder called 'chroma_db' to store our vectors
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create collection
    collection = client.get_or_create_collection(name="candidates")
    
    # If the database is empty, load the candidates from our JSON file
    if collection.count() == 0:
        print("Loading candidates into local vector database...")
        with open("data/candidates.json", "r") as f:
            candidates = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for c in candidates:
            # We create a dense text document for the embedding model to read
            doc = f"Title: {c['title']}. Skills: {', '.join(c['skills'])}. Bio: {c['bio']}. Experience: {c['years_experience']} years."
            documents.append(doc)
            metadatas.append(c)
            ids.append(c["id"])
            
        # ChromaDB automatically embeds these documents using its default free local model
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Database initialized and embeddings created successfully!")
    
    return collection

def evaluate_match_with_groq(jd, candidate_metadata):
    """Uses Groq to evaluate the candidate against the JD and output a score + explanation."""
    prompt = f"""
    You are an expert technical recruiter AI.
    
    Job Description:
    {jd}
    
    Candidate Profile:
    Name: {candidate_metadata['name']}
    Title: {candidate_metadata['title']}
    Skills: {', '.join(candidate_metadata['skills'])}
    Bio: {candidate_metadata['bio']}
    Experience: {candidate_metadata['years_experience']} years
    
    Task:
    1. Calculate a "Match Score" from 0 to 100 based strictly on how well the candidate's skills and experience align with the Job Description.
    2. Write a 2-3 sentence explanation for why you gave this score.
    
    Output format must be strictly JSON:
    {{"score": 85, "explanation": "Brief reason here..."}}
    """
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON-outputting evaluation agent. Only output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.1 # Low temperature for consistent scoring
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {"score": 0, "explanation": "Evaluation failed due to API error."}

def process_jd(jd_text, top_k=3):
    """Main pipeline: Find top candidates and evaluate them."""
    print("\n--- Starting JD Parsing & Candidate Discovery ---")
    collection = init_db()
    
    print(f"\nSearching for top {top_k} semantic matches...")
    # Query the vector DB
    results = collection.query(
        query_texts=[jd_text],
        n_results=top_k
    )
    
    scored_candidates = []
    
    # Loop through the top matches and have Groq evaluate them
    for metadata in results['metadatas'][0]:
        # Respect the candidate's status
        if not metadata['open_to_work']:
            print(f"Skipping {metadata['name']} - Not open to work.")
            continue 
            
        print(f"LLM Agent evaluating {metadata['name']}...")
        evaluation = evaluate_match_with_groq(jd_text, metadata)
        
        scored_candidates.append({
            "id": metadata["id"],
            "name": metadata["name"],
            "title": metadata["title"],
            "match_score": evaluation.get("score", 0),
            "match_explanation": evaluation.get("explanation", "No explanation provided."),
            "salary_expectation": metadata["current_salary_expectation"]
        })
        
    # Sort the final list by the Groq-generated match score (highest first)
    scored_candidates.sort(key=lambda x: x['match_score'], reverse=True)
    return scored_candidates

# Test block: This runs if you execute the script directly
if __name__ == "__main__":
    sample_job_description = """
    We are looking for a Senior AI Engineer with strong Python skills. 
    The ideal candidate should have hands-on experience building RAG applications 
    and working with Large Language Models. Minimum 4 years of experience required.
    """
    
    final_shortlist = process_jd(sample_job_description)
    
    print("\n--- Final Ranked Shortlist ---")
    print(json.dumps(final_shortlist, indent=2))