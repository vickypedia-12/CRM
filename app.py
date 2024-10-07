from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from retriever import generate_response, rag_manager
import os
import shutil
from dotenv import load_dotenv
import uvicorn
import json
from scrapy.crawler import CrawlerProcess

from scrapy.utils.log import configure_logging
from myproject.myproject.spiders.myspider import ContentExtractorSpider

load_dotenv()
app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    customer_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    history = [{"human": msg.content, "ai": ""} for msg in request.messages[:-1] if msg.role == "user"]
    for i, msg in enumerate(request.messages[:-1]):
        if msg.role == "assistant" and i > 0:
            history[i-1]["ai"] = msg.content
    
    query = request.messages[-1].content
    formatted_history = "\n".join([f"Human: {h['human']}\nAI: {h['ai']}" for h in history])
    
    response = generate_response(query, formatted_history, request.customer_id)
    return ChatResponse(response=response)

@app.post("/upload_document")
async def upload_document(customer_id: str = Body(...), file: UploadFile = File(...)):
    try:

        temp_customer_dir = f"Temp_Dataset_customer{customer_id}"
        os.makedirs(temp_customer_dir, exist_ok=True)

        file_path = os.path.join(temp_customer_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse(content={
            "message": f"File uploaded successfully to temporary directory for customer {customer_id}",
            "file_name": file.filename
        }, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/finish_documents")
async def finish_documents(customer_id: str = Body(...)):
    try:
        temp_customer_dir = f"Temp_Dataset_customer{customer_id}"
        final_customer_dir = f"Dataset_customer{customer_id}"
        if not os.path.exists(temp_customer_dir):
            raise HTTPException(status_code=400, detail=f"No temporary documents found for customer {customer_id}")
        
  
        os.makedirs(final_customer_dir, exist_ok=True)
        for filename in os.listdir(temp_customer_dir):
            shutil.move(os.path.join(temp_customer_dir, filename), os.path.join(final_customer_dir, filename))
        
   
        shutil.rmtree(temp_customer_dir)
        await update_dataset(customer_id, final_customer_dir)     
        return JSONResponse(content={
            "message": f"Documents finalized and dataset updated for customer {customer_id}"
        }, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_dataset")
async def update_dataset(customer_id: str = Body(...), new_directory: str = Body(...)):
    try:
        rag_manager.update_customer_dataset(customer_id, new_directory)
        return {"message": f"Dataset updated successfully for customer {customer_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/run_scrape")
async def run_scrape(start_url: str, customer_id: str):
    try:    
        # process = CrawlerProcess(settings={
        #     'FEED_FORMAT': 'json',
        #     'FEED_URI': 'output.json',
        #     'FEED_EXPORT_ENCODING': 'utf-8',
        # })
        # process.crawl(ContentExtractorSpider, start_url=start_url)
        # process.start()


        with open('output.json', 'r', encoding='utf-8') as json_file:
            scraped_data = json.load(json_file)

        text_content = ""
        for item in scraped_data:
            text_content += f"{item['content']}\n\n"

        customer_dir = f"Dataset_customer{customer_id}"
        os.makedirs(customer_dir, exist_ok=True)
        
        text_file_path = os.path.join(customer_dir, f"scraped_content_{customer_id}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text_content)

        new_directory = os.path.dirname(text_file_path)
        await update_dataset(customer_id, new_directory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/add_text_to_dataset")
async def add_text_to_dataset(customer_id: str = Body(...), user_text: str = Body(...)):
    try:
        
        customer_dir = f"Dataset_customer{customer_id}"
        os.makedirs(customer_dir, exist_ok=True)

        
        text_file_path = os.path.join(customer_dir, f"manual_input_{customer_id}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(user_text)

        await update_dataset(customer_id, customer_dir)

        return JSONResponse(content={
            "message": f"Text added successfully to dataset for customer {customer_id}",
            "text_file": f"manual_input_{customer_id}.txt"
        }, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_faq")
async def add_faq(customer_id: str = Body(...), question: str = Body(...), answer: str = Body(...)):
    try:
        customer_dir = f"Dataset_customer{customer_id}"
        os.makedirs(customer_dir, exist_ok=True)
        
        faq_file_path = os.path.join(customer_dir, f"faq_{customer_id}.json")
        
        
        if os.path.exists(faq_file_path):
            with open(faq_file_path, 'r', encoding='utf-8') as faq_file:
                faq_data = json.load(faq_file)
        else:
            faq_data = {}

        faq_data[question] = answer

        with open(faq_file_path, 'w', encoding='utf-8') as faq_file:
            json.dump(faq_data, faq_file, indent=4)

        await update_dataset(customer_id, customer_dir)

        return JSONResponse(content={
            "message": f"FAQ added successfully for customer {customer_id}",
            "faq_file": f"faq_{customer_id}.json"
        }, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)