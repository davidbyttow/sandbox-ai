import streamlit as st
import learn


chain = learn.ProfessorGPT.build()


plan_container = st.empty()


def generate_md():
    md = ""
    with st.spinner("Generating learning hub... (this may take up to a minute)"):
        out = chain(
            {
                "goal": st.session_state.goal,
                "reason": st.session_state.reason,
                "knowledge": st.session_state.knowledge,
            }
        )
        md = out.get("markdown", None)
        st.session_state.plan = md


def reset():
    st.session_state.pop("plan")


with st.sidebar:
    with st.form(key="learn_form"):
        st.text_input(label="What do you want to learn?", key="goal")
        st.text_input(label="Why do you want to learn this?", key="reason")
        st.text_input(label="What is your current level of knowledge?", key="knowledge")
        submit = st.form_submit_button(label="Generate", on_click=generate_md)

if "plan" in st.session_state:
    with plan_container:
        st.markdown(st.session_state.plan, unsafe_allow_html=True)
