import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text to math solver using google gemma 2 model")

with st.sidebar:
    groq_api_key= st.text_input('grop api key', value="", type='password')
    
    
    
    
if not groq_api_key:
    st.info("please enter your groq api key")
    st.stop()
    
llm=ChatGroq(model='gemma2-9b-it',groq_api_key=groq_api_key)

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name='wikipedia',
    func=wikipedia_wrapper.run,
    description='A tool for searching the Internet to find the vatious information on the topics mentioned'
)



math_wrapper=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name='calculator',
    func=math_wrapper.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="reasoning",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handling_parsing_errors=True
    
)

if 'messages'  not in st.session_state:
    st.session_state['messages']=[
        {"role":'assistant','content':"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
    
    
    
question=st.text_area("Enter youe question:","I have 8 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(input=st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")
    

