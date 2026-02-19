

▪ Overview / Introduction
▪ Large Language Models
▪ Reasoning
▪ Prompt and Context Engineering 
▪ Retrieval Augmented Generation 
▪ Memory
▪ Tools
▪ Design Patterns
▪ Communication
▪ Frameworks
▪ Orchestration
▪ Test, Evaluation and Benchmarking 
▪ Security, Trust and Identity

# 1 POPULAR AGENT DEFINITION 


Agent
▪ Ability to operate autonomously 
▪ Perceive the environment
▪ Persist over a prolonged time
▪ Adapt to change
▪ Create and pursue goals
▪ Rational agent: Act to achieve the best expected outcome



![](image/Pasted%20image%2020260218222718.png)


"An agent is anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators."  致动器

---


SAMPLE ARCHITECTURE
![](image/Pasted%20image%2020260218222819.png)

---

Weak Notion of Agency
▪ Ability to operate autonomously 
▪ Perceive the environment
▪ Persist over a prolonged time
▪ Adapt to change
▪ Create and pursue goals
▪ Rational agent: Act to achieve the best expected outcome


Strong Notion of Agency 
▪ Human-like properties
▪ Knowledge
▪ Belief
▪ Intention 
▪ Obligation


From a Reinforcement Learning Perspective
▪ Having goals relating to the state of the environment
▪ Being able to sense the state of the environment to some extent
▪ Having the ability to take actions that effect the state of the environment
▪ Learn from own experience


## 1.1 AGENT TECHNOLOGIES OF THE PAST AND PRESENT

Rule-based/Expert Systems (1970s-today)
if-then-else rules, symbolic AI
▪ Strength: Predictable, explainable, high-level reasoning
▪ Limitations: Hard to scale, failed with unexpected inputs, unable to learn,

Planning & Reasoning Agents (1980s-2000s)
Formal logic, Belief-Desire-Intention (BDI)
▪ Strength: Reasoning about goals, beliefs, and plans
▪ Limitations: Do not handle noisy data or natural language, computationally expensive

Reactive Agents (1980s-1990s)
Direct sensor to action mapping
Strength: Robust, fast, and works in unpredictable environments
Limitations: No knowledge, no world model, no reasoning, no planning, no state

Machine Learning-based Agents (1990s-today)
Using machine learning models (Bayesian networks, decisions trees, SVMs) to make decision and predict outcome
▪ Strength: Adapt to data (learn), better cope with uncertainty and probabilities
▪ Limitations: Extensive trials (trial & error), complex feature engineering, models do not generalize well

Dialog Systems & Assistants (2010s-2020)
Speech-Intent-Action-Speech Pipeline
▪ Strength: Good for well structured task descriptions, efficient and low latency
▪ Limitations: No deep reasoning, low performance for malformed task descriptions



# 2 MODERN AI AGENTS


 it can perceive ... understand the context of the circumstance. It can reason ... how to solve the problem, and it can plan an action ... and take action ...


1 LLM-powered Autonomous Agents
▪ LLM functions as the agent's brain
▪ Plan: Task Decomposition, Self-Reflection
▪ Memorize: Sensory, Short-Term, Long-Term 
▪ Make use of external tools

2 Degree of Agenticness
"the degree to which a system can adaptably achieve complex goals in complex environments with limited direct supervision."
▪ Goal complexity
▪ Environmental complexity
▪ Adaptability
▪ Independence
If a system exhibits a high degree of agenticness it is considered an agentic AI system


3 Characteristics of Increasing Agency
"Agency is the property, agent is the role, and agentic is the adjective"
▪ Underspecification: Degree to which a system can accomplish a goal without concrete specification of how the goal is to be accomplished
▪ Directness of impact: Degree to which a system's actions affect the world without a human in the loop
▪ Goal-directedness: Degree to which a system acts as if it is trained to achieve a quantifiable objective
▪ Long-term planning: Degree to which a system is designed to make decisions that are temporally dependent upon each other and/or make predictions over a long-time horizon

"Agentic AI systems are characterized by the ability to take actions which consistently contribute towards achieving goals over an extended period of time, without their behavior having been specified in advance"


规范不完整性：系统在缺乏达成目标的具体操作规范时，仍能实现该目标的程度。

影响直接性：系统的行动在没有人类介入的情况下，直接影响外部世界的程度。

目标导向性：系统的行为仿佛经过训练以追求某个可量化目标的程度。

长期规划性：系统被设计用于做出在时间上相互依赖的决策，和/或进行长期预测的程度。



4 LLM-powered AI Agents
=="AI agents are language model-powered entities able to plan and take actions to execute goals over multiple iterations [...] each agent is given a persona and access to a variety of tools that will help them accomplish their job either independently or as part of a team. Some agents also contain a memory component, where they can save and load information [...]"==


AI Agency
Agentic profiles based on core properties ▪ Autonomy
▪ Efficacy
▪ Goal Complexity
▪ Generality


AI Agent as a Software Entity
"[...] defined as autonomous software entities engineered for goal- directed task execution within bounded digital environments [...] defined by their ability to perceive structured or unstructured inputs, to reason over contextual information, and to initiate actions toward achieving specific objectives."
▪ Autonomy
▪ Task-Specificity
▪ Reactivity and Adaptation


## 2.1 COMPARED TO TRADITIONAL AGENTS


![](image/Pasted%20image%2020260218232401.png)


## 2.2 DEFINITIONS

What is AI Agent
=="An AI Agent is a system that leverages an LLM to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions to fulfill tasks."==


Agentic AI
"Agentic AI describes systems of multiple AI agents collaborating to achieve complex goals"


Modern AI Assistants
▪ Powered by LLMs
▪ Combines conversational ability with task assistance
▪ Understands complex and ambiguous language
▪ Supports multi-turn conversation
▪ Advanced context awareness



## 2.3 FROM BAISC LLM TO AGENTIC AI

![](image/Pasted%20image%2020260218232556.png)





# 3 PROPERTIES

AGENCY, AUTONOMY AND SELF-GOVERNANCE

## 3.1 Agency
"[...] the capability of [...] any entity to act independently and make choices [...]"

Decisional Authority
This refers to the power or ability to act and perform actions according to a chosen alternative or course of action. To possess the autonomy to evaluate different options and select the most appropriate action based on their internal decision-making processes, rather than being solely driven by external forces or predetermined rules.

Intentionality
Implies the existence of intentions, goals, or objectives that guide the actions and behavior of the system. To have a sense of purpose and to pursue specific objectives, adjusting actions and strategies as necessary to achieve those goals.

Responsibility
Is the answerability or accountability for the outcomes and consequences of one's actions. To be considered responsible for decisions and the impact of actions on the environment or other entities interacted with.



## 3.2 Autonomy
"[...] the extent 程度；范围，长度  to which an AI agent is designed to operate without user involvement"
自治，自治权；独立自主，自主权

"[...] 人工智能主体被设计为无需用户介入即可运行的程度"

Operational Autonomy
Refers to the degree of human operator disinvolvement at the AI agent's runtime, drawing inspiration from the conceptual focus of autonomy as independence from external factors.

Intentional Autonomy
The degree of AI agent goal-oriented involvement at runtime. An AI agent being both oriented towards a particular goal and having the knowledge of this goal. Not accidently achieving the goal but having an internal representation of the goal.

Shared Autonomy
Is the degree to which both the human operator and the AI agent are intentionally involved at runtime, existing as a composite of operational and intentional autonomy.

Non-deterministic Autonomy
Is the degree to which an AI agent's behavior is not specified prior to runtime. AI agents with high non-deterministic autonomy can decide the optimal path during runtime to reach a goal and adapt to novel situations.

Cognitive Autonomy
Is the degree to which the AI agent takes cognitive action, which broadly includes the ability to make and execute decisions, set goals, and plan based on environmental information.

Action Autonomy
Is defined as the degree to which the AI agent takes action. This includes acting autonomy, which is the freedom of the AI agent to move around in space and not be confined.


## 3.3 Self-governance
[...] ability of a system or entity to ==govern or control itself ==autonomously, without external direction or control."

"[...] 系统或实体在无需外部指导或控制的情况下，自主管理或操控自身的能力。"


Self-Organization
The ability of an AI agent to organize and structure its own internal processes, resources, and behavior without external intervention

Self-Regulation
The capability of an AI agent to monitor and adjust its own actions and outputs based on feedback from the environment or internal states, to ensure it operates within desired parameters or constraints

Self-Adaptation
The ability of an AI agent to modify its behavior, strategies, or decision-making processes in response to changes in the environment or its own internal conditions, to achieve its goals more effectively

Self-Optimization
The ability of an AI agent to continuously improve its performance, efficiency, or decision-making capabilities through learning, experience, or evolutionary processes

Self-Determination
The ability of an AI agent to set its own objectives, priorities, and courses of action based on its 
internal decision-making processes, without being entirely controlled by external forces


## 3.4 SUMMARY

Properties Characterizing an Agentic AI Setup
▪ Agency (decisional authority, intentionality, accountability/liability)
▪ Autonomy (operational, intentional, shared, cognitive, action)
▪ Self-Governance (self-organization, self-regulation, self-
adaptation, self-optimization, self-determination)
▪ Generality
▪ Reasoning
▪ "Reflectability"
▪ Determinism vs. non-determinism
▪ Single vs. Multi
▪ Short, long-term or no memory
▪ Tools vs. no tools
▪ Goal complexity
▪ Alignment with human values / organizational policies / law
▪ Human interface vs. no human interface
▪ Ability to create other agents (see action autonomy)
▪ Technical Roles: Controller, Verifier, Web Searcher, etc.
▪ Business Roles: Information, interaction or operational agent
▪ Skills/capabilities (knowledge retrieval, negotiating, coding/execution, coordination)
▪ "Identifiability"
▪ Reachability
▪ "Findability"
▪ Environment complexity
▪ Explainability
▪ Cooperativity
▪ "Perceptability" (via human input, sensors)
▪ Continuous perception, periodic perception,
▪ Human-triggered perception, externally-triggered perception, internally-triggered perception
▪ Ownership (private person, private entity, organization, not owned by anyone)
▪ Modalities (natural language text, images, videos, audio)
▪ Energy-consumption.
▪ Access to its execution environment ▪...

# 4 TAXONOMIES 分类

BASED ON INTERACTION SCHEME

![](image/Pasted%20image%2020260218235045.png)

## 4.1 BASED ON BUSINESS ROLE

Information Agent
▪ Document agents (extracting and organizing data from documents)
▪ Brower agents (assist with web surfing)
▪ Data agents (turning raw data into insights)
▪ Report agents (generating content)
▪ Unstructured agent (processing unstructured data)

Interaction Agent
▪ Conversational agents (providing client and employee support) ▪ Customer agents (streamlining customer support)

Operational Agent
▪ Decision agents (making business decisions)
▪ Data agents (combining data and tools for insights)
▪ Code agents (assisting with software development)
▪ Negotiating agents (negotiate shared resource usage)

Mobile Agents
▪ Monitoring agents (move through and observe a network) ▪ Digital twin agents (move with the physical twin)
▪ Personal assistants (move with principal)

Educational Agents
▪ Tutoring agents (explain concept, give hints, personalize)
▪ Pedagogical agents (detect, explain, and guide and motivate)
▪ Companion agents (virtual peer to support with language learning)
▪ Motivational agents (motivate)
▪ Legal advisory agents (explaining and contextualizing legal texts)
▪ Compliance agents (recommend alignment with compliance requirements)

## 4.2 BASED ON OPERATIONAL AUTONOMY

![](image/Pasted%20image%2020260218235141.png)



## 4.3 BASED ON MULTIPLE PROPERTIES


![](image/Pasted%20image%2020260218235238.png)


Properties
▪ Environmental interaction: Ability to perceive, understand, and manipulate the environment
▪ Goal directed behavior: Ability to form, understand, and pursue objectives
▪ Temporal coherence: Ability to maintain consistent operation over time through state awareness and memory
▪ Learning and adaptation: Capacity to improve its performance and adjust to new situation over time
▪ Autonomy: Ability to operate without constant external guidance




# 5 Big Picture 


Functional Perspective 
![](image/Pasted%20image%2020260218235343.png)


TECHNICAL PERSPECTIVE
![](image/Pasted%20image%2020260218235350.png)

WORKFLOW

![](image/Pasted%20image%2020260218235416.png)

