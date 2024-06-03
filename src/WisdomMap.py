#!/usr/bin/env python
# coding: utf-8

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import TokenTextSplitter
import os
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import sys
from langchain.llms import OpenAI

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
CORS(app, origins='*')

os.environ["OPENAI_API_KEY"]= "Key"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
#persist_directory = "test-db4"

from langchain_community.vectorstores import Chroma

db = Chroma(collection_name='collection-db4',persist_directory="test-db4",embedding_function=embeddings)
print('create DB instance',db)
print(db.get())
from langchain.prompts import PromptTemplate
def checkfrom_faiss(user_query):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})
    all_context = retriever.get_relevant_documents(user_query)
    print(all_context)
    context = """"""
    for i in range(3):
        context = context + '\n' + all_context[i].dict().get('page_content')
    return context


llm = OpenAI(temperature=0.2,seed = 123)


def make_question(keyword, other_word):
    question_templates = {
        "what": f"what is {other_word} with technical details from context?",
        "who" : f"which person or scientist is mentioned in relevance to {other_word} and how with technical details from context?",
        "elaborate" : f"Elaborate on {other_word} with technical details from context?",
        "pros": f" what are the pros of {other_word} with technical details from context?",
        "cons": f" what are the cons of {other_word} with technical details from context?",
        "extract" : f"Extract useful information about {other_word} with technical details from context?",
        "analogy" : f"Give an analogy for {other_word} with technical details from context if possible?",
        "explain" : f"Explain {other_word} with technical details from context?",
        "compare" : f"Compare {other_word}with other drugs with technical details from context?",
        "implications" : f"What are the implications of {other_word} with technical details from context?",
        "research" : f"What are some research done on {other_word} with technical details from context?",
        "concepts" : f"Mention few concepts related to {other_word} with technical details from context?",
        "controversy" : f"Mention few controversies related to {other_word} with technical details from context?",
        "significance" : f"what is the significance of {other_word} with technical details from context?",
        "interesting" : f"what is interesting about {other_word} with technical details from context?",
        "assessment": f"explain assessment under nursing process of {other_word} with technical details from context?",
        "diagnoses": f"explain diagnoses under nursing process of {other_word} with technical details from context.",
        "planning": f"explain planning under nursing process of {other_word} with technical details from context?",
        "implementation": f"explain implementation under nursing process of {other_word} with technical details from context?",
        "evaluation": f"Give evaluation under nursing process of {other_word} with technical details from context.",
    }

    if keyword in question_templates:
        return question_templates[keyword]
    if keyword =="Translate":
        a = translate_text()
    else:
        return "Invalid keyword."

# Example usage:
keyword = "explain"
other_word = "machine learning"
question = make_question(keyword, other_word)
print(question)


def main_openaicall(query):

    context = checkfrom_faiss(query)
    template = """You are a Question-Answering AI chatting with a Human about different processes of drug administration.
    Fetch data from the context and answer the question using only the context given in technical detail.Mention specific details from context.\
Refer the context and stick to original words from the context as much as possible. The context and the question is delimited by triple backticks.\
You provide lots of specific details from the context.\
If you do not know the answer to a question, truthfully say\
"I don't know. Question out of context.", don't try to make up an answer.\
Limit answer to 200 words.\

    Context:'''{context}'''

    Question: '''{query}'''

    Answer: """

    prompt = PromptTemplate(
    input_variables=["context","query"],
    template=template)
    llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
)
    output = llm_chain({'query': query, 'context': context})
    return output['text']



def main(question_word,root_word):
    question = make_question(question_word,root_word)
    print(question)
    ans = main_openaicall(question)
    return ans


@app.route('/api', methods=['GET'])
def api():
    #data = request.get_json()

    # Assuming 'root' and 'keyword' are keys in the JSON payload
    root = request.args.get('root')
    keyword = request.args.get('keyword')

    print('root ',root)
    print('keyword ',keyword)
    if root and keyword:
        print(root,keyword)
        answer = main(keyword,root)
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': 'Missing required parameters'}), 400

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)
     app.run(debug=False)


