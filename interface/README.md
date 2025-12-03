# Interface

## Setup

Before running the interface, make sure to do:

1. Install node
On MacOS/Linux
```bash
brew install node
```
After installation, confirm that everything is installed correctly:
```bash
node -v
npm -v
```

After confirming node is installed correctly:
```bash
cd ./interface/fronted
npm install
```

To run frontend only;
```bash
npm run dev
```
and open localhost http://localhost:5173/

Make sure python and pip is installed on your machine
Before running backed, install necessary packages
```bash
pip install fastapi
pip install uvicorn
pip install starlette
pip install pydantic
pip install "fastapi[standard]" motor python-dotenv
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
