#                     CLEAR CHAT
# =====================================================

st.sidebar.markdown("---")

if st.sidebar.button(
    "🗑️ Clear Chat",
    use_container_width=True
):

    st.session_state.messages = []

    st.rerun()


# =====================================================
#                    MAIN HEADER
# =====================================================

st.markdown(
    """
<div class="hero-card">
<h2>Welcome to your Smart PDF Assistant ✨</h2>
<p>Ask questions, explore your documents, and get accurate answers with source references and page numbers.</p>
</div>
""",
    unsafe_allow_html=True
)


# =====================================================
#                 DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            show_sources(message["sources"])


# =====================================================
#                    ASK QUESTION
# =====================================================

question = st.chat_input(
    "Ask anything about your PDFs..."
)


if question:

    # Display user question

    with st.chat_message("user"):

        st.markdown(question)

    # Retrieve relevant chunks

    docs = retrieve_chunks(
        question,
        selected_file
    )

    if not docs:

        answer = "I could not find this in the selected documents."

        sources = []

    else:

        # Generate answer

        with st.spinner("Generating answer..."):

            answer = ask_model(
                question,
                docs,
                st.session_state.messages
            )

            sources = build_sources(docs)

    # Display assistant answer

    with st.chat_message("assistant"):

        st.markdown(answer)

        if sources:

            show_sources(sources)

    # Save user message

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Save assistant message

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
