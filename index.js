import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient } from "@supabase/supabase-js";
//import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
//import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import * as dotenv from "dotenv";
import fs from "fs";
import { ChatOpenAI } from "@langchain/openai";

import { StringOutputParser } from "@langchain/core/output_parsers";

dotenv.config();

//import { PromptTemplate } from "langchain/prompts";
import { PromptTemplate } from "@langchain/core/prompts";

const openAIApiKey = process.env.OPENAI_API_KEY;

const embeddings = new OpenAIEmbeddings({ openAIApiKey });
const sbApiKey = process.env.SB_API_KEY;
const sbUrl = process.env.SB_URL;
const client = createClient(sbUrl, sbApiKey);

const vectorStore = new SupabaseVectorStore(embeddings, {
  client,
  tableName: "documents",
  queryName: "match_documents",
});

const retriever = vectorStore.asRetriever();

const llm = new ChatOpenAI({ openAIApiKey });

// A string holding the phrasing of the prompt
const standaloneQuestionTemplate =
  "generate a standalone question from this question:{wannaBeStandAloneQuestion}";

// A prompt created using PromptTemplate and the fromTemplate method
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);

// Take the standaloneQuestionPrompt and PIPE the model
const standaloneQuestionChain = standaloneQuestionPrompt
  .pipe(llm)
  .pipe(new StringOutputParser())
  .pipe(retriever);

// Await the response when you INVOKE the chain.
// Remember to pass in a question.
const response = await standaloneQuestionChain.invoke({
  wannaBeStandAloneQuestion:
    "Hi, I am Marcus, a collegue of yours.How are you? How are things? I had contact from Jenny the nurse at the hospital. I am working all days of the week, but not on monday. I was wondering if you are free when I am free for a yoga lesson",
});

console.log(response);

console.log("hello2");
