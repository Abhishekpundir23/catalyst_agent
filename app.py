__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import json
from rag_engine import process_jd
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables and initialize Groq
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Configure the Streamlit page
st.set_page_config(page_title="Catalyst Recruiter AI", page_icon="🚀", layout="wide")

# Initialize Session State (this keeps our data alive as we click around)
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []
if "active_candidate" not in st.session_state:
    st.session_state.active_candidate = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interest_score" not in st.session_state:
    st.session_state.interest_score = 0

# Sidebar Navigation
st.sidebar.title("Deccan AI Pipeline")
app_mode = st.sidebar.radio("Navigation", ["1. Discovery Engine", "2. Outreach Simulator", "3. Final Rankings"])

# -----------------------------------------
# PHASE 1: DISCOVERY ENGINE
# -----------------------------------------
if app_mode == "1. Discovery Engine":
    st.title("Phase 1: Candidate Discovery 🔍")
    st.write("Paste a Job Description below to search the vector database and generate Match Scores.")
    
    jd_input = st.text_area("Job Description:", "We are looking for a Senior AI Engineer with strong Python skills. Experience building RAG applications and working with Large Language Models. Minimum 4 years of experience required.", height=150)
    
    if st.button("Run AI Matcher", type="primary"):
        with st.spinner("Vectorizing JD and searching local database..."):
            st.session_state.shortlist = process_jd(jd_input)
        st.success("Matches found!")
    
    if st.session_state.shortlist:
        st.divider()
        st.subheader("Top Candidates")
        for cand in st.session_state.shortlist:
            with st.container():
                st.markdown(f"### {cand['name']} - Match: **{cand['match_score']}/100**")
                st.write(f"**Title:** {cand['title']} | **Expected Salary:** {cand['salary_expectation']}")
                st.info(f"**AI Reasoning:** {cand['match_explanation']}")
                
                if st.button(f"Select {cand['name']} for Outreach", key=cand['id']):
                    st.session_state.active_candidate = cand
                    # Reset chat history for the new candidate
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": f"Hi {cand['name']}, I'm the AI recruiter from Deccan. Your profile looks great for a role we have! Are you open to a quick chat about it?"}
                    ]
                    st.sidebar.success(f"Selected {cand['name']}. Click '2. Outreach Simulator' in the sidebar!")

# -----------------------------------------
# PHASE 2: OUTREACH SIMULATOR
# -----------------------------------------
elif app_mode == "2. Outreach Simulator":
    st.title("Phase 2: Conversational Engagement 💬")
    
    if not st.session_state.active_candidate:
        st.warning("Please select a candidate from the Discovery Engine first.")
    else:
        cand = st.session_state.active_candidate
        st.markdown(f"**Simulating conversation with:** {cand['name']}")
        st.caption("You will act as the candidate responding to the AI Recruiter.")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input box for the user (acting as the candidate)
        if prompt := st.chat_input(f"Type {cand['name']}'s response here..."):
            # Add user message to state and display it
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate AI Recruiter response
            with st.chat_message("assistant"):
                with st.spinner("Agent is typing..."):
                    messages = [
                        {"role": "system", "content": f"You are a professional AI recruiter interviewing {cand['name']}. Ask 1 question at a time to assess their genuine interest in the role and their salary expectations. Keep responses brief (1-2 sentences)."}
                    ] + st.session_state.chat_history
                    
                    response = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.5
                    )
                    ai_reply = response.choices[0].message.content
                    st.write(ai_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
        
        st.divider()
        # Button to end chat and calculate score
        if st.button("End Chat & Calculate Interest Score", type="primary"):
            with st.spinner("Analyzing conversation sentiment and intent..."):
                eval_prompt = f"""
                Analyze this conversation between an AI recruiter and a candidate.
                Score the candidate's genuine interest from 0 to 100 based on their enthusiasm, availability, and alignment.
                Return ONLY valid JSON in this exact format: {{"interest_score": 85, "reason": "brief explanation"}}
                
                Conversation History:
                {json.dumps(st.session_state.chat_history, indent=2)}
                """
                
                eval_response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a JSON-only evaluation agent. Only output valid JSON."},
                        {"role": "user", "content": eval_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                result = json.loads(eval_response.choices[0].message.content)
                score = result.get("interest_score", 0)
                reason = result.get("reason", "No reason provided.")
                
                st.success(f"Interest Score Calculated: {score}/100")
                st.write(f"**Analysis:** {reason}")
                
                # Save the score back to the candidate in our shortlist
                for c in st.session_state.shortlist:
                    if c['id'] == cand['id']:
                        c['interest_score'] = score
                        c['final_score'] = (c['match_score'] + score) / 2

# -----------------------------------------
# PHASE 3: FINAL RANKINGS
# -----------------------------------------
elif app_mode == "3. Final Rankings":
    st.title("Phase 3: Recruiter Dashboard 🏆")
    
    if not st.session_state.shortlist:
        st.warning("No data yet. Run the discovery engine first.")
    else:
        # Prepare list for display, defaulting interest score if chat hasn't happened
        display_list = []
        for c in st.session_state.shortlist:
            c_copy = c.copy()
            if 'interest_score' not in c_copy:
                c_copy['interest_score'] = "Pending Outreach"
                c_copy['final_score'] = c_copy['match_score'] / 2 # Penalize for no outreach
            display_list.append(c_copy)
            
        # Sort by final combined score
        display_list.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        for i, c in enumerate(display_list):
            st.markdown(f"### Rank #{i+1}: {c['name']}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Match Score (RAG)", c['match_score'])
            with col2:
                st.metric("Interest Score (Chat)", c['interest_score'])
            with col3:
                final_val = c.get('final_score', 0)
                formatted_final = f"{final_val:.1f}" if isinstance(final_val, (int, float)) else final_val
                st.metric("FINAL COMBINED SCORE", formatted_final)
                
            st.divider()