import os
import streamlit as st
import cohere
import requests
from crewai import Agent, Task, Crew, Process

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatCohere

from langchain.tools import DuckDuckGoSearchRun
from crewai_tools import SeleniumScrapingTool, ScrapeWebsiteTool
from duckduckgo_search import DDGS

# Ensure essential environment variables are set
cohere_api_key = os.getenv('COHERE_API_KEY')
if not cohere_api_key:
    raise EnvironmentError("COHERE_API_KEY is not set in environment variables")
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in environment variables")

# Initialize API clients
co = cohere.Client(cohere_api_key)

# Jina.ai API key
jina_api_key = os.getenv('JINA_API_KEY')

def fetch_content(url):
    try:
        response = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {jina_api_key}"}
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching content: {e}")
        return f"Error fetching content: {e}"

def search_results(search_query: str) -> dict:
    """Performs a web search to gather and return a collection of search results."""
    results = DDGS().text(search_query, max_results=5, timelimit='m')
    results_list = [{"title": result['title'], "snippet": result['body'], "link": result['href']} for result in results]
    return results_list

def web_scrapper(url: str, topic: str) -> str:
    """Extracts and reads the content of a specified link and generates a summary on a specific topic."""
    content = fetch_content(url)
    prompt = f"Generate a summary of the following content on the topic ## {topic} ### \n\nCONTENT:\n\n" + content
    response = co.chat(
        model='command-r',
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
        openai_llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        cohere_llm = ChatCohere(temperature=0, cohere_api_key=cohere_api_key)
        selected_llm = openai_llm

        researcher = Agent(
            role='Researcher',
            goal=f'Search and Collect detailed information on topic ## {topic} ##',
            backstory="You are a meticulous researcher, skilled at navigating vast amounts of information to extract essential insights on any given topic.",
            allow_delegation=False,
            llm=selected_llm,
            tools=[
                DuckDuckGoSearchRun(),
                SeleniumScrapingTool(),
                ScrapeWebsiteTool()
            ],
            verbose=True
        )

        editor = Agent(
            role='Editor',
            goal=f'Compile and refine the information into a comprehensive report on topic ## {topic} ##',
            backstory="As an expert editor, you specialize in transforming raw data into clear, engaging reports.",
            allow_delegation=False,
            llm=selected_llm,
            verbose=True
        )

        research_task = Task(
            description=f"Research the topic: {topic}. Use search tools to find relevant information, then use web scraping tools to gather more detailed data from the most promising sources.",
            agent=researcher
        )

        edit_task = Task(
            description=f"Review and refine the research on {topic}. Create a well-structured report with an introduction, main body, and conclusion. Ensure all information is accurate and properly cited.",
            agent=editor
        )

        crew = Crew(
            agents=[researcher, editor],
            tasks=[research_task, edit_task],
            process=Process.sequential
        )

        result = crew.kickoff()
        return result
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="CrewAI Research Tool", page_icon="🔍", layout="wide")
    st.title("CrewAI Research Tool")

    topic = st.text_input("Enter Topic", placeholder="Type here...")

    if st.button("Start Research"):
        if topic:
            with st.spinner("Researching... This may take a few minutes."):
                result = kickoff_crew(topic)
                st.markdown(result)
        else:
            st.warning("Please enter a topic before starting the research.")

if __name__ == "__main__":
    main()
