# Interface

## Setup

Before running the interface, make sure to do:

```bash
npm install
pip install uvicorn fastapi
chmod +x run.sh
```

## Running the Application

To run the frontend and backend server, you can run:

```bash
./run.sh
```

## Testing

If you would like to test alert functionality, you can also start the testing server:

```bash
cd ./testing
uvicorn ml_server:app --reload --port 8080 --host 127.0.0.1
```
