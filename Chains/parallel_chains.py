from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

model2 = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the follwoing text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text\n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge both notes and quiz into a single document \n {notes}{quiz} ",
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser
})

sequential_chain = prompt3 | model1 | parser

chain= parallel_chain | sequential_chain

text="""tbh i keep seeing everyone online calling “AI Agents” basically anything that uses GPT-4 inside an automation flow… and that’s just not how it works. like yeah, you’re calling your fancy automation “agents” but most of the time you’re just slapping GPT on top of if-this-then-that logic

let’s be real. n8n is amazing. i use it daily. i love it. you can build insane integrations, workflows, triggers, api calls, webhooks, data pipelines… but that alone doesn’t make your automation an ai agent

for context: i’m a software engineer with 8+ years of experience, i work full time building ai automations and teaching others how to build real ai agents. and yeah, i use n8n heavily. but i also know where its limits are

if you actually break down what AI Agents are in most definitions, you’ll find 7 core types. depending on which one you’re trying to build, n8n can fully handle some, partially handle others, and for a few it’s simply not designed for that job

so here’s how i see it, based on actual builds i’ve done:

reactive agents — these are the simplest form. input comes in, agent reacts. no state, no memory, no long-term reasoning. faq bots for example. you take user input, send it to gpt-4 or claude, return the answer. super easy to build fully inside n8n. honestly this is what most people today call “ai agents” in SaaS but technically speaking it’s just automation with LLM calls on top

deliberative agents — now you’re building systems that actually try to model the world a little bit. like pulling traffic, weather, or historical data and making decisions based on that. this you can actually build in n8n, if you wire everything manually. you connect external apis, store data in supabase or postgres, run reasoning inside gpt-4 calls. but you’re writing the full logic flow. n8n isn’t deciding by itself

goal-based agents — these work toward specific objectives. like a sales agent qualifying leads, adapting its approach, trying to close a deal. in n8n you can build partial flows for this: store lead state, query pinecone or qdrant for embeddings, inject that into prompts. but you still have to handle the whole decision logic yourself. n8n doesn’t track goals or adjust behavior automatically over time

utility-based agents — these don’t just follow goals but optimize across multiple variables for best outcomes. like dynamic pricing models reacting to demand, inventory, competition. here n8n simply doesn’t have the tools. you’ll need external ML models, optimization engines, forecasting algorithms. n8n might orchestrate calls but doesn’t handle the core optimization logic

learning agents — these actually improve over time by learning from experience. like a support bot fine-tuning itself using past conversations and user feedback. n8n can absolutely help orchestrate data collection, prep datasets, kick off fine-tuning jobs. but the learning system itself fully lives outside of n8n. the learning logic is not inside your workflow builder

hybrid agents — these combine both planning and instant reactions. autonomous vehicles are a classic example. they plan full routes but react immediately to obstacles. real-time, multi-layered reasoning. this kind of agent behavior is not something you can simulate inside n8n. workflows aren’t designed for real-time closed-loop reasoning

multi-agent systems — here you’ve got multiple agents coordinating, negotiating, working together. like agents handling different parts of a supply chain. n8n can absolutely help orchestrate external systems but true agent-to-agent coordination requires pub/sub layers, message brokers, distributed systems. n8n isn’t built to be that communication layer

so where does n8n actually fit?

if you combine it with a few external tools you can get surprisingly far depending on the problem you're solving. i typically use supabase or postgres for state, pinecone or qdrant for semantic memory, gpt-4o or claude for reasoning, langchain planner or crewai for planning, and sometimes simulate loops in n8n by simply calling the workflow again with updated state. for very basic multi-agent coordination i’ve used supabase realtime or redis pubsub

bottom line: n8n is insanely good for orchestration. you can build very useful agent-like behaviors that deliver huge business value. but fully autonomous ai agents — the kind that manage their own state, reason independently, learn and adapt, coordinate between agents — those systems live mostly outside of n8n’s core capabilities

and that’s where i keep seeing people overselling what n8n can do. yes you can plug in llms, yes you can store state externally, yes you can simulate loops. but you’re not building real autonomous agents — you’re building advanced automation flows that simulate some agent behaviors, which is still extremely valuable. but let’s not confuse one thing with the other

curious to hear how others see this — will n8n ever build native agent capabilities? or will it always stay in orchestration territory?"""
result=chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()