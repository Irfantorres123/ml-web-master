import uvicorn


if __name__ == "__main__":
    uvicorn.run("api:app", root_path='/api', host="0.0.0.0", port=8000, reload=True)