from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from retriever import generate_response, rag_manager
import os
import shutil
from dotenv import load_dotenv
import uvicorn
import asyncio
import tempfile
import json
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from myproject.myproject.spiders.myspider import SubdirectorySpider
from myproject.myproject.spiders.content_extractor import ContentExtractorSpider
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
        # Create temporary customer directory if it doesn't exist
        temp_customer_dir = f"Temp_Dataset_customer{customer_id}"
        os.makedirs(temp_customer_dir, exist_ok=True)
        
        # Save the uploaded file to the temporary directory
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
        
        # Check if temporary directory exists
        if not os.path.exists(temp_customer_dir):
            raise HTTPException(status_code=400, detail=f"No temporary documents found for customer {customer_id}")
        
        # Move files from temporary to final directory
        os.makedirs(final_customer_dir, exist_ok=True)
        for filename in os.listdir(temp_customer_dir):
            shutil.move(os.path.join(temp_customer_dir, filename), os.path.join(final_customer_dir, filename))
        
        # Remove the temporary directory
        shutil.rmtree(temp_customer_dir)
        
        # Update the dataset
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
        
async def run_spider(spider, spider_kwargs=None):
    spider_kwargs = spider_kwargs or {}
    configure_logging()  
    
    process = CrawlerProcess(get_project_settings())
    
    process.crawl(spider, **spider_kwargs)
    process.start()

    return {"message": "Spider completed successfully"}

async def run_subdirectory_spider(start_url: str):
    try:
        spider_kwargs = {"start_urls": [start_url]}
        result = await run_spider(SubdirectorySpider, spider_kwargs)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_content_extractor_spider():
    try:
        result = await run_spider(ContentExtractorSpider)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/run_full_scrape_workflow")
async def run_full_scrape_workflow(start_url: str):
    try:
    
        await run_subdirectory_spider(start_url)
        
        if os.path.exists('urls3.json'):
            os.rename('urls3.json', 'urls.json')
        else:
            raise HTTPException(status_code=500, detail="urls3.json not found after running subdirectory spider")
        
        await run_content_extractor_spider()
        
        return JSONResponse(content={"message": "Full scrape workflow completed successfully"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)