# Backend Interview Guide — Expense Tracker

---

## 1. Project Overview

I built an Expense Tracker application using the **MERN stack** (MongoDB, Express.js, React, Node.js). The backend is a RESTful API that handles all the expense data — creating, reading, updating, and deleting expenses. The frontend (React) communicates with this backend through HTTP requests.

**Folder structure:**
```
backend/
├── server.js              → Main entry point, starts the server
├── models/
│   └── Expense.js         → Defines the data structure (schema)
├── routes/
│   └── expenses.js        → API endpoints (controller logic)
├── package.json           → Project dependencies
```

---

## 2. server.js — The Entry Point

`server.js` is the main file that initializes and starts the backend server. It does 4 key things:

### 2.1 Importing Libraries

```js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();
```

- **Express** — A web framework for Node.js. It simplifies creating HTTP servers, defining routes, and handling requests. Without Express, I would have to use Node's raw `http` module and manually parse URLs, headers, and request bodies.

- **Mongoose** — An ODM (Object Data Modeling) library for MongoDB. It lets me define schemas (data blueprints) and gives me easy methods like `.find()`, `.save()`, `.findByIdAndUpdate()` to interact with the database.

- **CORS** — My React frontend runs on `localhost:3000` and the backend on `localhost:5000`. Browsers block requests between different origins by default. The CORS middleware adds headers that tell the browser it's safe to allow these cross-origin requests.

- **dotenv** — Loads environment variables from a `.env` file. This keeps sensitive data like database URLs and port numbers out of the code.

### 2.2 Middleware Setup

```js
app.use(cors());
app.use(express.json());
```

- `cors()` — Runs on every incoming request and adds CORS headers so the frontend can talk to the backend.

- `express.json()` — Parses incoming JSON request bodies. When the frontend sends `{ "title": "Groceries", "amount": 500 }`, this middleware converts the raw JSON string into a JavaScript object at `req.body`. Without this, `req.body` would be `undefined`.

### 2.3 MongoDB Connection

```js
mongoose.connect('mongodb://localhost:27017/expense-tracker', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('MongoDB connected'))
.catch(err => console.log(err));
```

- Connects to a local MongoDB database called `expense-tracker`.
- `.then()` logs a success message when the connection is established.
- `.catch()` catches and logs any connection errors (e.g., if MongoDB isn't running).
- `useNewUrlParser` and `useUnifiedTopology` avoid deprecation warnings.

### 2.4 Route Mounting

```js
app.use('/api/expenses', require('./routes/expenses'));
```

- All requests starting with `/api/expenses` are forwarded to the routes file.
- This is **modular routing** — it keeps `server.js` clean and makes it easy to add more routes later (e.g., `/api/users`).

### 2.5 Starting the Server

```js
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

- Starts the HTTP server on port 5000 and begins listening for requests.

---

## 3. MongoDB & Mongoose Model — models/Expense.js

### 3.1 Why MongoDB?

I chose MongoDB because:
- Expense data is simple and doesn't need complex joins like SQL databases.
- MongoDB stores data as JSON-like documents, which pairs naturally with JavaScript/Node.js.
- It's flexible — I can easily add or remove fields without database migrations.

### 3.2 The Schema

```js
const ExpenseSchema = new mongoose.Schema({
  title:       { type: String, required: true },
  amount:      { type: Number, required: true },
  category:    { type: String, required: true },
  date:        { type: Date,   required: true },
  description: { type: String, default: '' }
}, {
  timestamps: true
});
```

A schema defines what each expense document looks like in the database:

- **title** (String, required) — The name of the expense like "Rent" or "Groceries". If missing, Mongoose rejects the save and returns an error.

- **amount** (Number, required) — The monetary value. Mongoose automatically rejects non-numeric values like `"abc"`, protecting data integrity.

- **category** (String, required) — Groups expenses like "Food", "Transport". Useful for filtering and analytics on the frontend.

- **date** (Date, required) — When the expense occurred. Used for sorting by date in the API.

- **description** (String, optional) — Extra notes. Defaults to an empty string so the frontend never gets `undefined`.

- **timestamps: true** — Automatically creates `createdAt` and `updatedAt` fields on every document. Useful for auditing and debugging.

### 3.3 The Model

```js
module.exports = mongoose.model('Expense', ExpenseSchema);
```

- Creates a model named `Expense` from the schema.
- Mongoose automatically creates a collection called `expenses` in MongoDB (lowercase, pluralized).
- The model provides database methods: `.find()`, `.save()`, `.findByIdAndUpdate()`, `.findByIdAndDelete()`.

### 3.4 Sample Document in MongoDB

```json
{
  "_id": "65f8a2b3c4d5e6f7a8b9c0d1",
  "title": "Groceries",
  "amount": 2500,
  "category": "Food",
  "date": "2026-03-23T00:00:00.000Z",
  "description": "Weekly vegetables",
  "createdAt": "2026-03-23T07:30:00.000Z",
  "updatedAt": "2026-03-23T07:30:00.000Z"
}
```

---

## 4. Routes / Controller — routes/expenses.js

This file contains all the API endpoints. It acts as the **controller** — it receives requests, talks to the database through the model, and sends responses back. I implemented full CRUD operations following REST principles.

### 4.1 GET /api/expenses — Read All Expenses

```js
router.get('/', async (req, res) => {
  try {
    const expenses = await Expense.find().sort({ date: -1 });
    res.json(expenses);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});
```

- **Purpose:** Fetches all expenses from the database.
- **`Expense.find()`** — Gets all documents from the `expenses` collection.
- **`.sort({ date: -1 })`** — Sorts newest first. `-1` means descending order.
- **`res.json(expenses)`** — Sends the data as JSON to the frontend.
- **Error:** Returns `500` (Server Error) because a read failure is usually a database problem.
- **Frontend use:** Called when the page loads to display the expense list.

### 4.2 POST /api/expenses — Create New Expense

```js
router.post('/', async (req, res) => {
  try {
    const expense = new Expense(req.body);
    const savedExpense = await expense.save();
    res.status(201).json(savedExpense);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});
```

- **Purpose:** Creates a new expense in the database.
- **`new Expense(req.body)`** — Creates an Expense instance from the frontend form data.
- **`.save()`** — Validates against the schema and inserts into MongoDB. If a required field is missing, it throws an error.
- **`res.status(201)`** — Returns `201 Created` (REST best practice, more specific than `200 OK`).
- **Error:** Returns `400` (Bad Request) because create failures are usually due to invalid data from the client.
- **Frontend use:** Called when the user submits the "Add Expense" form.

### 4.3 PUT /api/expenses/:id — Update Expense

```js
router.put('/:id', async (req, res) => {
  try {
    const updatedExpense = await Expense.findByIdAndUpdate(
      req.params.id, req.body, { new: true }
    );
    res.json(updatedExpense);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});
```

- **Purpose:** Updates an existing expense by its ID.
- **`/:id`** — URL parameter. The `:id` captures the expense's MongoDB `_id` from the URL.
- **`req.params.id`** — Extracts the ID value from the URL.
- **`findByIdAndUpdate(id, data, options)`** — Finds the document by `_id` and applies the updates.
- **`{ new: true }`** — Returns the updated document instead of the old one. Without this, the frontend would get stale data.
- **Error:** Returns `400` (Bad Request) for invalid update data.
- **Frontend use:** Called when the user edits an expense and saves.

### 4.4 DELETE /api/expenses/:id — Delete Expense

```js
router.delete('/:id', async (req, res) => {
  try {
    await Expense.findByIdAndDelete(req.params.id);
    res.json({ message: 'Expense deleted' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});
```

- **Purpose:** Removes an expense from the database.
- **`findByIdAndDelete()`** — Finds and deletes the document in one operation.
- **`res.json({ message: 'Expense deleted' })`** — Sends a confirmation back.
- **Error:** Returns `500` (Server Error) for database failures.
- **Frontend use:** Called when the user clicks the "Delete" button.

### 4.5 Error Handling Pattern

Every route uses `try/catch`:
```js
try {
  // database operation
} catch (error) {
  res.status(400 or 500).json({ message: error.message });
}
```

- `400` = Client error (bad data from frontend) — used for POST and PUT.
- `500` = Server error (database problem) — used for GET and DELETE.
- This prevents the entire server from crashing on a single bad request.

---

## 5. How the Complete Request Flow Works

```
Step  What Happens
----  ----------------------------------------------------------------
1     User clicks "Add Expense" on the React frontend
2     Frontend sends POST /api/expenses with JSON body to port 5000
3     Express server receives the request
4     cors() middleware allows the cross-origin request
5     express.json() middleware parses the JSON body into req.body
6     Express matches the URL to the route handler in routes/expenses.js
7     Route handler creates a new Expense model and calls .save()
8     Mongoose validates the data and sends a query to MongoDB
9     MongoDB saves the document and returns the result
10    Route handler sends a JSON response back with status 201
11    Frontend receives the response and updates the UI
```

---

## 6. RESTful API Summary

REST means mapping HTTP methods to database operations on a resource:

```
Method    URL                    Action         Status Code
------    ---                    ------         -----------
GET       /api/expenses          Read all       200 OK
POST      /api/expenses          Create new     201 Created
PUT       /api/expenses/:id      Update one     200 OK
DELETE    /api/expenses/:id      Delete one     200 OK
```

**REST principles I followed:**

1. **Resource-based URLs** — `/api/expenses` represents the resource. No verbs like `/getExpenses`.
2. **HTTP methods define actions** — GET reads, POST creates, PUT updates, DELETE removes.
3. **Stateless** — Each request is independent. The server doesn't remember previous requests.
4. **JSON format** — All data is sent and received as JSON.
5. **Proper status codes** — 201 for created, 400 for bad data, 500 for server errors.
6. **Modular code** — Server config, data models, and routes are in separate files.

---

## 7. Interview Questions & Answers

### Q1. What is the tech stack of your backend?
> Node.js as the runtime, Express.js as the web framework, MongoDB as the NoSQL database, and Mongoose as the ODM library.

### Q2. Why did you choose MongoDB over MySQL?
> Expense data is simple and doesn't need complex joins. MongoDB's flexible document structure makes it easy to store JSON-like expense objects. It also pairs naturally with Node.js since both use JavaScript/JSON.

### Q3. What does server.js do?
> It's the entry point. It initializes Express, sets up middleware (CORS and JSON parsing), connects to MongoDB, mounts the API routes, and starts the server on port 5000.

### Q4. What is middleware and what middleware did you use?
> Middleware are functions that run on every request before it reaches the route handler. I used `cors()` to allow cross-origin requests from my React frontend, and `express.json()` to parse JSON request bodies.

### Q5. Why did you use Mongoose instead of the native MongoDB driver?
> Mongoose provides schema validation, type casting, and easy query methods. Without it, I'd have to manually validate every field and write raw MongoDB queries.

### Q6. What does `required: true` do in your schema?
> It makes the field mandatory. If the frontend sends data without that field, Mongoose throws a validation error and the API returns a 400 Bad Request response.

### Q7. What does `timestamps: true` do?
> It automatically adds `createdAt` and `updatedAt` fields to every document. I don't have to manually track when records were created or modified.

### Q8. What is `{ new: true }` in findByIdAndUpdate?
> By default, `findByIdAndUpdate` returns the old document (before the update). `{ new: true }` tells it to return the updated version instead, which is what the frontend needs to display.

### Q9. Why do you use `status(201)` for POST instead of `200`?
> 201 specifically means "Created" — it tells the client that a new resource was successfully created. This is a REST best practice. 200 just means "OK" which is less informative.

### Q10. How do you handle errors in your APIs?
> Every route handler is wrapped in a `try/catch` block. If the database operation fails, the catch block sends back an appropriate status code (400 for client errors, 500 for server errors) with the error message as JSON. This prevents the server from crashing.

### Q11. What is `express.Router()` and why did you use it?
> `express.Router()` creates a mini-router for handling a group of related routes. I used it to keep all expense-related routes in a separate file (`routes/expenses.js`) instead of cluttering `server.js`. This makes the code modular and maintainable.

### Q12. What does `async/await` do in your routes?
> Database operations take time. `async/await` makes these operations non-blocking — the server can handle other requests while waiting for MongoDB to respond, instead of freezing up.

### Q13. What would happen if you removed `express.json()` middleware?
> `req.body` would be `undefined` in all POST and PUT routes. The server wouldn't be able to read any data sent from the frontend forms, and creating/updating expenses would break completely.

### Q14. How is your API different from a non-RESTful API?
> A non-RESTful API might use URLs like `/getExpenses`, `/deleteExpense`, `/createExpense`. My RESTful API uses one resource URL `/api/expenses` and lets the HTTP method (GET, POST, PUT, DELETE) determine the action. This is cleaner, more standardized, and follows industry conventions.

### Q15. How would you add authentication to this backend?
> I would add a JWT (JSON Web Token) based authentication system — create a User model, add login/register routes, generate a token on login, and create a middleware that checks the token on every protected route before allowing access.
