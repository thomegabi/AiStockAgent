

#IMPORTANDO AS LIBS

import os
import json
from datetime import datetime

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import yfinance as yf

import streamlit as st


# In[3]:


#CRIANDO YAHOO_TOOL
def fetch_stock_price(ticket):
  stock = yf.download(ticket, start="2023-08-08", end="2024-08-08") #AAPL é o codigo de estoque da apple
  return stock

yahoo_finance_tool = Tool(
  name = "Yahoo Finance Tool",
  description = "Fetches stock prices for {ticket} from the last year about an specific company from Yahoo Finance API",
  func = lambda ticket: fetch_stock_price(ticket)
)

#IMPORTANDO OPEN_AI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")

#CRIANDO O AGENTE DE ANALISE DE PREÇO DE MERCADO
stockPriceAnalyst = Agent(
  role = "Senior stock price analyst",
  goal = "Find the {ticket} stock price and analyses trends",
  backstory = """ You are highly experienced in analysing prices of an specific stock and do predicitons 
    about its future value and behavior """,
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  tools = [yahoo_finance_tool],
  allow_delegation = False
)


#CRIANDO A TAREFA DO AGENTE
getStockPrice = Task(
  description = "Analyse stock of {ticket} price history and create a trend analysis of up, down or sideways",
  expected_output = """ Specify the current trend stock price - up, down or sideways
  eg. stock= 'AAPL, price UP'
  """,
  agent = stockPriceAnalyst
)


#IMPORTANDO A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

#CRIANDO AGENTE DE ANALISE DE NOTICIAS
newsAnalyst = Agent(
  role = "Stock news analyst",
  goal = """Create a short sumary with the latest market news about the stock {ticket} company. Specify the current trend - up, down, sideways
   with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
  backstory = """ 
    You are a highly experienced in analyzing the market trends and news, and have tracked assets form more then 10 years.
   
    You are also master level analysts in the tradicional markets and have deep understanding of human psychology.

    You understand news, theis tittles and information, but you look at those with a health dose of skepticism.
    You consider also the source of the news articles.
     """,
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  tools = [search_tool],
  allow_delegation = False
)


#CRIANDO A TASK DO AGENTE DE NOTICIAS

#current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

getNews = Task(
  description = """
  Take the stock and always include BTC to it (if not request).

  Use the search tool to search each one individualy.

  the current date is {datetime.now()}.

  Compose the results into a helpfull report.
  """,
  expected_output = """ 
  A summary of the overall marcket and one sentence summary for each request asset.
  Include a fear/greed score for each asset based on the news. Use format:
  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
  agent = newsAnalyst 
)

#CRIA O ANALISTA FINAL
stockAnalystWriter = Agent(
  role = "Senior Stock Analyst Writer",
  goal = """ Analyse the trends prices and write an insightfull compelling and informative 3 paragraph long newsletter baser on the stock report and proce trend. """,
  backstory = """ 
    You are widely accepted as the best stock analyst in the market.
    You understand complex concepts and create compelling stories and narratives that resonate wider audiences

    You understand macro factors and combine multiple theories - eg. cycle theoyy and fundamental analyses.
    You are able to hold multiple opinions when analyzing anything
     """,
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  allow_delegation = True,
)


writeAnalysis = Task(
  description = """
    Use the stock price trend and stock news repoty to create an analyses and write teh newsletter about the {ticket} company that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
  """,
  expected_output ="""
    An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. ItShould contain:

    - 3 bullets executive summary
    - Introduction - ser the overall picture and spice up the interest
    - main part provides the meat of the analysis including news summary and dear/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways
  """,
  agent = stockAnalystWriter,
  context = [getStockPrice, getNews]
)


crew = Crew(
  agents = [stockAnalystWriter, stockPriceAnalyst, newsAnalyst],
  tasks= [getNews, getStockPrice, writeAnalysis],
  verbose = True,
  process = Process.hierarchical,
  full_output = True,
  share_crew = False,
  manager_llm = llm,
  max_iter = 15,
)


results = crew.kickoff(inputs={
  'ticket': 'AAPL',
})


results['final_output']


with st.sidebar:
  st.header('Enter the Stock to Research')

  with st.form(key='research_form'):
    topic = st.text_input("Select the ticket")
    submit_button = st.form_submit_button(label = "Run Research")

  if submit_button:
    if not topic:
      st.error("Please fill the ticket field")
    else: 
      results = crew.kickoff(inputs={'ticket': topic})

      st.subheader("Results of your research:")
      st.write(results['final_output'])


