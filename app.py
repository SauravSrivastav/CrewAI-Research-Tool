import streamlit as st
import cohere
import requests
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from duckduckgo_search import DDGS

# Function to initialize session state
def init_session_state():
    if 'cohere_api_key' not in st.session_state:
        st.session_state.cohere_api_key = ""
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'jina_api_key' not in st.session_state:
        st.session_state.jina_api_key = ""

def fetch_content(url):
    try:
        response = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {st.session_state.jina_api_key}"}
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching content: {e}")
        return f"Error fetching content: {e}"

def search_results(search_query: str) -> list:
    results = DDGS().text(search_query, max_results=5, timelimit='m')
    results_list = [{"title": result['title'], "snippet": result['body'], "link": result['href']} for result in results]
    return results_list

def web_scrapper(url: str, topic: str) -> str:
    content = fetch_content(url)
    prompt = f"Generate a summary of the following content on the topic ## {topic} ### \n\nCONTENT:\n\n" + content
    co = cohere.Client(st.session_state.cohere_api_key)
    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=1000,
        temperature=0.4,
    )
    summary_response = f"""###
    Summary:
    {response.generations[0].text}
    
    URL: {url}
    ###
    """
    return summary_response

def research_topic(topic: str) -> str:
    try:
        groq_llm = ChatGroq(temperature=0, groq_api_key=st.session_state.groq_api_key, model_name="llama3-70b-8192")
        cohere_llm = ChatCohere(
            temperature=0,
            cohere_api_key=st.session_state.cohere_api_key,
            model_name="command"
        )

        tools = [
            Tool(
                name="Search",
                func=lambda q: str(search_results(q)),
                description="useful for when you need to search for information on the internet"
            ),
            Tool(
                name="WebScraper",
                func=lambda url_topic: web_scrapper(url_topic.split(',')[0], url_topic.split(',')[1]),
                description="useful for when you need to get detailed information from a specific webpage"
            )
        ]

        memory = ConversationBufferMemory(memory_key="chat_history")

        agent = initialize_agent(
            tools, 
            groq_llm, 
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
            memory=memory,
            verbose=True
        )

        result = agent.run(f"Research the topic: {topic}. Provide a comprehensive report including an introduction, main body, and conclusion. Cite all sources used.")
        return result
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="AI Research Tool", page_icon="🔍")
    st.title("AI Research Tool")

    init_session_state()

    # API Key Input Section
    st.header("API Key Configuration")
    st.write("Please enter your API keys below. If you don't have them, you can obtain them from the following sources:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Cohere API Key")
        st.session_state.cohere_api_key = st.text_input("Cohere API Key", value=st.session_state.cohere_api_key, type="password")
        st.markdown("[Get Cohere API Key](https://dashboard.cohere.ai/)")
    
    with col2:
        st.subheader("Groq API Key")
        st.session_state.groq_api_key = st.text_input("Groq API Key", value=st.session_state.groq_api_key, type="password")
        st.markdown("[Get Groq API Key](https://console.groq.com/)")
    
    with col3:
        st.subheader("Jina AI API Key")
        st.session_state.jina_api_key = st.text_input("Jina AI API Key", value=st.session_state.jina_api_key, type="password")
        st.markdown("[Get Jina AI API Key](https://jina.ai/)")

    # Research Section
    st.header("Research Topic")
    topic = st.text_input("Enter Topic", placeholder="Type here...")
    
    if st.button("Start Research"):
        if topic and st.session_state.cohere_api_key and st.session_state.groq_api_key and st.session_state.jina_api_key:
            with st.spinner("Researching... This may take a few minutes."):
                result = research_topic(topic)
                st.markdown(result)
        else:
            st.warning("Please enter a topic and ensure all API keys are provided before starting the research.")

if __name__ == "__main__":
    main()
