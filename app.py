from utils import TalkToRepo
import streamlit as st

def main():
    st.set_page_config(page_title="TalkToRepo", page_icon="🧊")
    if "chain" not in st.session_state:
        st.session_state.chain = TalkToRepo()
    if "history" not in st.session_state:
        st.session_state.history = []
    with st.sidebar:
        st.markdown("### Enter the repository link")
        repo_link = st.text_input("Repository Link")
        if st.button("Submit"):
            with st.spinner("Building Engine...."):
                st.session_state.chain.build_engine(repo_link)

    st.title("TalkToRepo")
    st.markdown("##### Ask your questions about the code in the repository")
    for chat in st.session_state.history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

    question=st.chat_input("Ask your question")
    if question:
        with st.chat_message("human"):
            st.markdown(question)
        res=st.session_state.chain.get_response(question)
        with st.chat_message("assistant"):
            st.markdown(res)

        st.session_state.history.append({"role":"human", "message":question})
        st.session_state.history.append({"role":"assistant", "message":res})

if __name__ == "__main__":
    main()