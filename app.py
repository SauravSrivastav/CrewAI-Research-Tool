import streamlit as st
import cohere
import requests
from crewai import Agent, Task, Crew, Process

from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere

from crewai_tools import tool
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

@tool('DuckDuckGoSearchResults')
def search_results(search_query: str) -> dict:
    results = DDGS().text(search_query, max_results=5, timelimit='m')
    results_list = [{"title": result['title'], "snippet": result['body'], "link": result['href']} for result in results]
    return results_list

@tool('WebScrapper')
def web_scrapper(url: str, topic: str) -> str:
    content = fetch_content(url)
    prompt = f"Generate a summary of the following content on the topic ## {topic} ### \n\nCONTENT:\n\n" + content
    co = cohere.Client(st.session_state.cohere_api_key)
    response = co.chat(
        model='command-r-plus',
        message=prompt,
        temperature=0.4,
        max_tokens=1000,
        chat_history=[],
        prompt_truncation='AUTO'
    )
    summary_response = f"""###
    Summary:
    {response.text}
    
    URL: {url}
    ###
    """
    return summary_response

def kickoff_crew(topic: str) -> dict:
    try:
        groq_llm_70b = ChatGroq(temperature=0, groq_api_key=st.session_state.groq_api_key, model_name="llama3-70b-8192")
        cohere_llm = ChatCohere(
            temperature=0,
            cohere_api_key=st.session_state.cohere_api_key,
            model_name="command-r-plus"
        )

        selected_llm = groq_llm_70b

        researcher = Agent(
            role='Researcher',
            goal=f'Search and Collect detailed information on topic ## {topic} ##',
            tools=[search_results, web_scrapper],
            llm=selected_llm,
            backstory=(
                "You are a meticulous researcher, skilled at navigating vast amounts of information to extract essential insights on any given topic. "
                "Your dedication to detail ensures the reliability and thoroughness of your findings. "
                "With a strategic approach, you carefully analyze and document data, aiming to provide accurate and trustworthy results."
            ),
            allow_delegation=False,
            max_iter=15,
            max_rpm=20,
            memory=True,
            verbose=True
        )

        editor = Agent(
            role='Editor',
            goal=f'Compile and refine the information into a comprehensive report on topic ## {topic} ##',
            llm=selected_llm,
            backstory=(
                "As an expert editor, you specialize in transforming raw data into clear, engaging reports. "
                "Your strong command of language and attention to detail ensure that each report not only conveys essential insights "
                "but is also easily understandable and appealing to diverse audiences. "
            ),
            allow_delegation=False,
            max_iter=5,
            max_rpm=15,
            memory=True,
            verbose=True
        )

        research_task = Task(
            description=(
                f"Use the DuckDuckGoSearchResults tool to collect initial search snippets on ## {topic} ##. "
                f"If more detailed searches are required, generate and execute new queries related to ## {topic} ##. "
                "Subsequently, employ the WebScrapper tool to delve deeper into significant URLs identified from the snippets, extracting further information and insights. "
                "Compile these findings into a preliminary draft, documenting all relevant sources, titles, and links associated with the topic. "
                "Ensure high accuracy throughout the process and avoid any fabrication or misrepresentation of information."
            ),
            expected_output=(
                "A structured draft report about the topic, featuring an introduction, a detailed main body organized by different aspects of the topic, and a conclusion. "
                "Each section should properly cite sources, providing a thorough overview of the information gathered."
            ),
            agent=researcher
        )

        edit_task = Task(
            description=(
                "Review and refine the initial draft report from the research task. Organize the content logically to enhance information flow. "
                "Verify the accuracy of all data, correct discrepancies, and update information to ensure it reflects current knowledge and is well-supported by sources. "
                "Improve the report's readability by enhancing language clarity, adjusting sentence structures, and maintaining a consistent tone. "
                "Include a section listing all sources used, formatted as bullet points following this template: "
                "- title: url'."
            ),
            expected_output=(
                f"A polished, comprehensive report on topic ## {topic} ##, with a clear, professional narrative that accurately reflects the research findings. "
                "The report should include an introduction, an extensive discussion section, a concise conclusion, and a well-organized source list. "
                "Ensure the document is grammatically correct and ready for publication or presentation."
            ),
            agent=editor,
            context=[research_task]
        )

        crew = Crew(
            agents=[researcher, editor],
            tasks=[research_task, edit_task],
            process=Process.sequential,
        )

        result = crew.kickoff()
        return result
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="CrewAI Research Tool", page_icon="🔍")
    st.title("CrewAI Research Tool")

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
                result = kickoff_crew(topic)
                st.markdown(result)
        else:
            st.warning("Please enter a topic and ensure all API keys are provided before starting the research.")

if __name__ == "__main__":
    main()
