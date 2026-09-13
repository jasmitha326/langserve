import os
from datetime import datetime

import streamlit as st
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Study Planner Agent",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Student Study Planner Agent")
st.write("Enter your subjects, exam date, study hours, and difficult subjects.")


# ---------------------------------------------------------
# GEMINI API KEY
# ---------------------------------------------------------
# Recommended:
# Set your Gemini API key as an environment variable:
# GEMINI_API_KEY="your_api_key"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is not set. "
        "Please add your Gemini API key as an environment variable."
    )
    st.stop()


# ---------------------------------------------------------
# TOOLS
# ---------------------------------------------------------
@tool
def calculate_days(exam_date: str) -> str:
    """Calculate the number of days remaining until the exam date."""
    try:
        exam = datetime.strptime(exam_date, "%Y-%m-%d").date()
        today = datetime.today().date()
        days = (exam - today).days

        if days < 0:
            return "The exam date has already passed."
        if days == 0:
            return "The exam is today."

        return f"{days} days are remaining until the exam."
    except ValueError:
        return "Please provide the exam date in YYYY-MM-DD format."


@tool
def prioritize_subjects(
    subjects: str,
    difficult_subjects: str
) -> str:
    """Prioritize difficult subjects first."""
    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    difficult_list = {
        s.strip().lower()
        for s in difficult_subjects.split(",")
        if s.strip()
    }

    priority = [
        subject for subject in subject_list
        if subject.lower() in difficult_list
    ]

    remaining = [
        subject for subject in subject_list
        if subject.lower() not in difficult_list
    ]

    ordered = priority + remaining

    if not ordered:
        return "No subjects were provided."

    return "Priority order: " + " → ".join(ordered)


@tool
def calculate_study_hours(
    hours_per_day: float,
    days: int
) -> str:
    """Calculate total available study hours."""
    if hours_per_day <= 0 or days <= 0:
        return "Study hours and days must be greater than zero."

    total = hours_per_day * days
    return f"Total available study time: {total:.1f} hours."


@tool
def create_study_plan(
    subjects: str,
    hours_per_day: float,
    days: int,
    difficult_subjects: str
) -> str:
    """Create a simple day-wise study plan."""
    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]

    if not subject_list:
        return "Please provide at least one subject."

    if hours_per_day <= 0 or days <= 0:
        return "Study hours and days must be greater than zero."

    difficult_list = {
        s.strip().lower()
        for s in difficult_subjects.split(",")
        if s.strip()
    }

    priority = [
        subject for subject in subject_list
        if subject.lower() in difficult_list
    ]

    remaining = [
        subject for subject in subject_list
        if subject.lower() not in difficult_list
    ]

    ordered_subjects = priority + remaining

    plan = []
    for day in range(1, days + 1):
        subject = ordered_subjects[(day - 1) % len(ordered_subjects)]

        if subject.lower() in difficult_list:
            focus = "Difficult subject – give extra focus and practice."
        else:
            focus = "Study concepts and revise important topics."

        plan.append(
            f"Day {day}: {subject} | "
            f"{hours_per_day:.1f} hours | {focus}"
        )

    return "\n".join(plan)


tools = [
    calculate_days,
    prioritize_subjects,
    calculate_study_hours,
    create_study_plan
]


# ---------------------------------------------------------
# GEMINI MODEL
# ---------------------------------------------------------
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ---------------------------------------------------------
# AGENT
# ---------------------------------------------------------
agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a Student Study Planner Agent. "
        "Your job is ONLY to help students create study plans, "
        "organize subjects, calculate study time, prioritize difficult subjects, "
        "and prepare revision schedules. "
        "Use the available tools whenever they are useful. "
        "Give answers in simple, clear language. "
        "For questions unrelated to student study planning, "
        "say exactly: "
        "'I am not authorized to answer questions outside of student study planning.'"
    )
)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
subjects = st.text_input(
    "Subjects",
    placeholder="Example: Machine Learning, Data Mining, OS, Web Programming"
)

exam_date = st.date_input(
    "Exam Date"
)

hours_per_day = st.number_input(
    "Study hours per day",
    min_value=0.5,
    max_value=24.0,
    value=6.0,
    step=0.5
)

difficult_subjects = st.text_input(
    "Difficult subjects",
    placeholder="Example: Machine Learning, Data Mining"
)


# ---------------------------------------------------------
# GENERATE PLAN
# ---------------------------------------------------------
if st.button("Generate Study Plan", type="primary"):

    if not subjects.strip():
        st.warning("Please enter your subjects.")
        st.stop()

    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]

    days = (exam_date - datetime.today().date()).days

    if days < 1:
        st.warning("Please select a future exam date.")
        st.stop()

    query = f"""
Create a study plan for me.

Subjects: {", ".join(subject_list)}
Exam date: {exam_date.strftime("%Y-%m-%d")}
Days remaining: {days}
Study hours per day: {hours_per_day}
Difficult subjects: {difficult_subjects if difficult_subjects.strip() else "None"}

Use the available tools to calculate the study time, prioritize subjects,
and create a practical day-wise study plan.

Give the final answer in a simple format with:
1. Days remaining
2. Total study hours
3. Subject priority
4. Day-wise study plan
5. Short revision tips
"""

    with st.spinner("Creating your study plan..."):
        try:
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })

            final_message = result["messages"][-1]

            st.success("Study plan generated!")
            st.markdown("### 📅 Your Study Plan")

            if hasattr(final_message, "content"):
                content = final_message.content

                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = "\n".join(text_parts)

                st.write(content)
            else:
                st.write(final_message)

        except Exception as e:
            st.error(f"Error while running the agent: {e}")
