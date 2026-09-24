import os
import re
import json
from collections import defaultdict
from typing import TypedDict, Optional

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

CATEGORIES = [
    "Food", "Groceries", "Transport", "Utilities", "Shopping",
    "Entertainment", "Healthcare", "Education", "Other"
]

KEYWORDS = {
    "Food": [
        "restaurant", "food", "meal", "lunch", "dinner", "breakfast",
        "snack", "pizza", "burger", "cafe", "coffee"
    ],
    "Groceries": [
        "milk", "rice", "vegetable", "vegetables", "fruit", "fruits",
        "grocery", "groceries", "dal", "atta", "bread", "egg", "eggs"
    ],
    "Transport": [
        "uber", "ola", "cab", "taxi", "bus", "metro", "train",
        "flight", "fuel", "petrol", "diesel", "transport"
    ],
    "Utilities": [
        "electricity", "power", "water", "gas", "internet", "wifi",
        "mobile", "phone", "recharge", "utility"
    ],
    "Shopping": [
        "shirt", "shoes", "clothes", "shopping", "amazon", "flipkart",
        "cleaning", "item", "items", "electronics"
    ],
    "Entertainment": [
        "movie", "cinema", "netflix", "spotify", "game", "gaming",
        "concert", "entertainment"
    ],
    "Healthcare": [
        "doctor", "medicine", "medical", "hospital", "pharmacy",
        "health", "tablet"
    ],
    "Education": [
        "book", "books", "course", "college", "school", "education",
        "exam", "tuition", "training"
    ]
}


class ExpenseState(TypedDict, total=False):
    text: str
    budget: Optional[float]
    items: list
    total: float
    analysis: dict
    report: dict


def clean_amount(value):
    try:
        if isinstance(value, (int, float)):
            return float(value)

        value = str(value).replace(",", "")
        match = re.search(r"\d+(?:\.\d+)?", value)
        return float(match.group()) if match else 0.0
    except Exception:
        return 0.0


def categorize_local(name):
    text = name.lower()

    for category, words in KEYWORDS.items():
        if any(word in text for word in words):
            return category

    return "Other"


def local_parse(text):
    items = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^(.*?)(?:[-:=₹$]\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
            line
        )

        if not match:
            continue

        name = match.group(1).strip(" -:=₹$")
        amount = clean_amount(match.group(2))

        if name and amount > 0:
            items.append({
                "name": name,
                "amount": amount,
                "category": categorize_local(name)
            })

    return {
        "items": items,
        "total": round(sum(x["amount"] for x in items), 2)
    }


def get_gemini():
    if not API_KEY or ChatGoogleGenerativeAI is None:
        return None

    try:
        return ChatGoogleGenerativeAI(
            model=MODEL,
            google_api_key=API_KEY,
            temperature=0
        )
    except Exception:
        return None


def extract_with_gemini(text):
    llm = get_gemini()

    if not llm:
        return None

    prompt = f"""
You are an expense extraction system.

Extract only genuine expense items from the supplied text.

Rules:
- Never invent an amount.
- Never estimate missing prices.
- Ignore dates, invoice numbers, GST numbers, phone numbers,
  quantities and addresses unless they clearly represent money spent.
- Preserve the supplied amounts.
- Categorize each item using only these categories:
  Food, Groceries, Transport, Utilities, Shopping,
  Entertainment, Healthcare, Education, Other.
- Calculate total from extracted items.

Return ONLY valid JSON:

{{
  "items": [
    {{
      "name": "",
      "amount": 0,
      "category": ""
    }}
  ],
  "total": 0
}}

Expense text:
{text}
"""

    try:
        response = llm.invoke(prompt)
        raw = response.content

        if isinstance(raw, list):
            raw = "".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                for x in raw
            )

        raw = str(raw).strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        valid_items = []

        for item in data.get("items", []):
            name = str(item.get("name", "")).strip()
            amount = clean_amount(item.get("amount", 0))
            category = str(item.get("category", "Other"))

            if not name or amount <= 0:
                continue

            if category not in CATEGORIES:
                category = "Other"

            valid_items.append({
                "name": name,
                "amount": amount,
                "category": category
            })

        if not valid_items:
            return None

        total = round(sum(x["amount"] for x in valid_items), 2)

        return {
            "items": valid_items,
            "total": total
        }

    except Exception:
        return None


def extract_expenses(state):
    result = extract_with_gemini(state["text"])

    if result is None:
        result = local_parse(state["text"])

    state["items"] = result["items"]
    state["total"] = result["total"]

    return state


def categorize(state):
    items = []

    for item in state.get("items", []):
        category = item.get("category", "Other")

        if category not in CATEGORIES:
            category = categorize_local(item["name"])

        items.append({
            "name": item["name"],
            "amount": round(float(item["amount"]), 2),
            "category": category
        })

    state["items"] = items
    state["total"] = round(sum(x["amount"] for x in items), 2)

    return state


def analyze(state):
    items = state.get("items", [])
    total = state.get("total", 0)

    categories = defaultdict(float)

    for item in items:
        categories[item["category"]] += item["amount"]

    category_distribution = {
        key: round(value, 2)
        for key, value in sorted(
            categories.items(),
            key=lambda x: x[1],
            reverse=True
        )
    }

    top_items = sorted(
        items,
        key=lambda x: x["amount"],
        reverse=True
    )[:5]

    observations = []

    if total:
        observations.append(
            f"Total recorded spending is ₹{total:,.2f}."
        )

    if category_distribution:
        category, amount = next(iter(category_distribution.items()))
        percentage = round((amount / total) * 100, 1)

        observations.append(
            f"{category} is the largest category at "
            f"₹{amount:,.2f}, representing {percentage}% "
            f"of recorded spending."
        )

    if top_items:
        observations.append(
            f"The highest individual expense is "
            f"{top_items[0]['name']} at "
            f"₹{top_items[0]['amount']:,.2f}."
        )

    if len(items) >= 3:
        observations.append(
            f"{len(items)} individual expenses were identified."
        )

    return {
        "category_distribution": category_distribution,
        "high_cost_items": top_items,
        "observations": observations
    }


def generate_suggestions(state):
    analysis = analyze(state)

    items = state.get("items", [])
    total = state.get("total", 0)
    categories = analysis["category_distribution"]

    suggestions = []

    if categories:
        category, amount = next(iter(categories.items()))

        if total and amount / total >= 0.40:
            suggestions.append(
                f"{category} represents a large share of the "
                f"recorded spending. Review those expenses first "
                f"for potentially optional purchases."
            )

    for category in ["Shopping", "Entertainment", "Food"]:
        amount = categories.get(category, 0)

        if amount > 0:
            suggestions.append(
                f"Review ₹{amount:,.2f} spent on "
                f"{category.lower()} for purchases that may be optional."
            )

    if len(items) >= 3:
        suggestions.append(
            "Compare the largest individual expenses before "
            "trying to reduce smaller essential purchases."
        )

    if not suggestions:
        suggestions.append(
            "The supplied data is limited. Track more expenses "
            "before making significant spending changes."
        )

    budget = state.get("budget")
    budget_data = None

    if budget is not None and budget > 0:
        remaining = round(budget - total, 2)
        percentage = round((total / budget) * 100, 1)

        if total > budget:
            status = "Over budget"
        elif total >= budget * 0.9:
            status = "Near budget"
        else:
            status = "Within budget"

        budget_data = {
            "budget": round(budget, 2),
            "spent": round(total, 2),
            "remaining": remaining,
            "percentage_used": percentage,
            "status": status
        }

    state["analysis"] = analysis

    state["report"] = {
        "total": total,
        "items": items,
        "category_distribution": categories,
        "high_cost_items": analysis["high_cost_items"],
        "observations": analysis["observations"],
        "saving_opportunities": suggestions,
        "budget": budget_data
    }

    return state


def build_graph():
    workflow = StateGraph(ExpenseState)

    workflow.add_node("extract_expenses", extract_expenses)
    workflow.add_node("categorize", categorize)
    workflow.add_node("analyze", analyze)
    workflow.add_node("generate_suggestions", generate_suggestions)

    workflow.add_edge(START, "extract_expenses")
    workflow.add_edge("extract_expenses", "categorize")
    workflow.add_edge("categorize", "analyze")
    workflow.add_edge("analyze", "generate_suggestions")
    workflow.add_edge("generate_suggestions", END)

    return workflow.compile()


expense_graph = build_graph()


def answer_chat(question, report):
    question = question.lower().strip()

    items = report.get("items", [])
    categories = report.get("category_distribution", {})
    total = report.get("total", 0)

    if not items:
        return "There are no analyzed expenses yet."

    if "most" in question or "highest" in question:
        category = max(categories, key=categories.get)
        return (
            f"You are spending the most on {category}: "
            f"₹{categories[category]:,.2f}."
        )

    if "largest expense" in question or "costliest" in question:
        item = max(items, key=lambda x: x["amount"])
        return (
            f"Your largest individual expense is "
            f"{item['name']} at ₹{item['amount']:,.2f}."
        )

    for category in CATEGORIES:
        if category.lower() in question:
            amount = categories.get(category, 0)
            return f"You spent ₹{amount:,.2f} on {category}."

    if "total" in question or "spent" in question:
        return f"Your recorded spending is ₹{total:,.2f}."

    if "reduce" in question or "save" in question:
        return " ".join(report.get("saving_opportunities", [])[:3])

    if "budget" in question and report.get("budget"):
        budget = report["budget"]

        return (
            f"You have spent ₹{budget['spent']:,.2f} of "
            f"₹{budget['budget']:,.2f}, using "
            f"{budget['percentage_used']}% of the budget. "
            f"Remaining: ₹{budget['remaining']:,.2f}."
        )

    return (
        f"You recorded ₹{total:,.2f} across {len(items)} expenses. "
        f"Ask about your total, largest expense, categories, "
        f"budget, or ways to reduce spending."
    )


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Smart Expense Analyzer</title>

<style>
*{box-sizing:border-box}

body{
margin:0;
font-family:Arial,sans-serif;
color:#f5f7fb;
min-height:100vh;
background:
radial-gradient(circle at 10% 10%,#293d68 0,transparent 35%),
radial-gradient(circle at 90% 90%,#43265d 0,transparent 35%),
#090d16;
}

.container{
max-width:1100px;
margin:auto;
padding:35px 20px;
}

header{
text-align:center;
margin-bottom:30px;
}

h1{
font-size:38px;
margin:0 0 10px;
}

.subtitle{
color:#aeb8cb;
line-height:1.5;
}

.card{
background:rgba(255,255,255,.07);
border:1px solid rgba(255,255,255,.12);
border-radius:20px;
padding:24px;
margin-bottom:20px;
backdrop-filter:blur(14px);
box-shadow:0 15px 45px rgba(0,0,0,.25);
}

textarea{
width:100%;
min-height:190px;
resize:vertical;
background:rgba(0,0,0,.25);
border:1px solid rgba(255,255,255,.15);
border-radius:14px;
padding:16px;
color:white;
font-size:15px;
outline:none;
}

input{
width:230px;
padding:12px;
margin-top:10px;
border-radius:10px;
border:1px solid #444;
background:#101522;
color:white;
}

button{
border:0;
border-radius:12px;
padding:13px 22px;
margin-top:15px;
cursor:pointer;
font-weight:bold;
background:#7658ff;
color:white;
}

button:hover{
opacity:.88;
}

.grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
gap:16px;
}

.stat{
background:rgba(255,255,255,.06);
border-radius:16px;
padding:20px;
}

.stat small{
color:#9da8bd;
}

.stat strong{
display:block;
font-size:26px;
margin-top:7px;
}

.row{
display:flex;
justify-content:space-between;
gap:15px;
padding:11px 0;
border-bottom:1px solid rgba(255,255,255,.08);
}

ul{
line-height:1.8;
}

.chat{
display:flex;
gap:10px;
}

.chat input{
flex:1;
width:auto;
margin:0;
}

.hidden{
display:none;
}

.error{
color:#ff9292;
margin-top:12px;
}

.status{
display:inline-block;
margin-top:10px;
padding:6px 10px;
border-radius:8px;
background:rgba(124,92,255,.18);
color:#cfc5ff;
}

@media(max-width:600px){
h1{font-size:29px}
.container{padding:20px 12px}
.chat{flex-direction:column}
.chat input{width:100%}
}
</style>
</head>

<body>

<div class="container">

<header>
<h1>SMART EXPENSE ANALYZER</h1>

<div class="subtitle">
AI-powered bill extraction, spending analysis and practical
saving insights
</div>
</header>

<div class="card">

<h2>Analyze Your Expenses</h2>

<p class="subtitle">
Paste shopping, restaurant, grocery, travel or utility expenses.
</p>

<textarea id="expenseText"
placeholder="Milk 60
Rice 450
Vegetables 320
Snacks 280
Cleaning items 500"></textarea>

<br>

<label>Monthly Budget (optional)</label><br>

<input
id="budget"
type="number"
min="0"
placeholder="15000">

<br>

<button onclick="analyze()">Analyze Expenses</button>

<div id="error" class="error"></div>

</div>

<div id="result" class="hidden">

<div class="grid">

<div class="stat">
<small>TOTAL</small>
<strong id="total">₹0</strong>
</div>

<div class="stat">
<small>ITEMS</small>
<strong id="itemCount">0</strong>
</div>

<div class="stat">
<small>TOP CATEGORY</small>
<strong id="topCategory">-</strong>
</div>

</div>

<div id="budgetBox" class="card hidden">

<h2>BUDGET</h2>

<div class="grid">

<div class="stat">
<small>BUDGET</small>
<strong id="budgetAmount">₹0</strong>
</div>

<div class="stat">
<small>USED</small>
<strong id="budgetUsed">0%</strong>
</div>

<div class="stat">
<small>REMAINING</small>
<strong id="remaining">₹0</strong>
</div>

</div>

<div id="budgetStatus" class="status"></div>

</div>

<div class="card">

<h2>CATEGORIES</h2>

<div id="categories"></div>

</div>

<div class="card">

<h2>TOP EXPENSES</h2>

<div id="topExpenses"></div>

</div>

<div class="card">

<h2>INSIGHTS</h2>

<ul id="observations"></ul>

</div>

<div class="card">

<h2>SAVING OPPORTUNITIES</h2>

<ul id="suggestions"></ul>

</div>

<div class="card">

<h2>EXPENSE ASSISTANT</h2>

<div class="chat">

<input
id="question"
placeholder="Where am I spending the most?">

<button onclick="chat()">Ask</button>

</div>

<p id="chatAnswer"></p>

</div>

</div>
</div>

<script>

let currentReport=null;

function money(value){
return "₹"+Number(value).toLocaleString("en-IN",{
minimumFractionDigits:2,
maximumFractionDigits:2
});
}

async function analyze(){

const text=document.getElementById("expenseText").value;
const budget=document.getElementById("budget").value;
const error=document.getElementById("error");

error.textContent="";

if(!text.trim()){
error.textContent="Please enter some bill or expense information.";
return;
}

try{

const response=await fetch("/analyze",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({
text:text,
budget:budget
})
});

const data=await response.json();

if(!response.ok){
error.textContent=data.error||"Unable to analyze expenses.";
return;
}

currentReport=data;
render(data);

}catch(errorObject){
error.textContent="Unable to connect to the server.";
}

}

function render(data){

document.getElementById("result").classList.remove("hidden");

document.getElementById("total").textContent=money(data.total);

document.getElementById("itemCount").textContent=data.items.length;

const categories=data.category_distribution||{};

document.getElementById("topCategory").textContent=
Object.keys(categories)[0]||"-";

const categoryDiv=document.getElementById("categories");

categoryDiv.innerHTML="";

Object.entries(categories).forEach(([name,value])=>{

categoryDiv.innerHTML+=`
<div class="row">
<span>${name}</span>
<strong>${money(value)}</strong>
</div>`;

});

const topDiv=document.getElementById("topExpenses");

topDiv.innerHTML="";

data.high_cost_items.forEach(item=>{

topDiv.innerHTML+=`
<div class="row">
<span>
${item.name}
<small>(${item.category})</small>
</span>
<strong>${money(item.amount)}</strong>
</div>`;

});

fillList("observations",data.observations);

fillList("suggestions",data.saving_opportunities);

if(data.budget){

document.getElementById("budgetBox")
.classList.remove("hidden");

document.getElementById("budgetAmount")
.textContent=money(data.budget.budget);

document.getElementById("budgetUsed")
.textContent=data.budget.percentage_used+"%";

document.getElementById("remaining")
.textContent=money(data.budget.remaining);

document.getElementById("budgetStatus")
.textContent=data.budget.status;

}else{

document.getElementById("budgetBox")
.classList.add("hidden");

}

document.getElementById("result")
.scrollIntoView({behavior:"smooth"});

}

function fillList(id,items){

document.getElementById(id).innerHTML=
items.map(item=>`<li>${item}</li>`).join("");

}

async function chat(){

if(!currentReport){
return;
}

const question=
document.getElementById("question").value.trim();

if(!question){
return;
}

try{

const response=await fetch("/chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({
question:question,
report:currentReport
})
});

const data=await response.json();

document.getElementById("chatAnswer").textContent=
data.answer||data.error;

}catch(error){
document.getElementById("chatAnswer").textContent=
"Unable to connect to the server.";
}

}

</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze_route():
    try:
        data = request.get_json(force=True)

        text = str(data.get("text", "")).strip()

        if not text:
            return jsonify({
                "error": "Expense text is required."
            }), 400

        budget_value = data.get("budget")

        if budget_value in ("", None):
            budget = None
        else:
            budget = float(budget_value)

            if budget <= 0:
                return jsonify({
                    "error": "Budget must be greater than zero."
                }), 400

        result = expense_graph.invoke({
            "text": text,
            "budget": budget
        })

        report = result.get("report", {})

        if not report.get("items"):
            return jsonify({
                "error": (
                    "No expense items could be identified. "
                    "Try inputs such as 'Milk 60' or 'Rice ₹450'."
                )
            }), 400

        return jsonify(report)

    except ValueError:
        return jsonify({
            "error": "Please enter a valid numeric budget."
        }), 400

    except Exception:
        return jsonify({
            "error": "Unable to process the expense data."
        }), 500


@app.route("/chat", methods=["POST"])
def chat_route():
    try:
        data = request.get_json(force=True)

        question = str(data.get("question", "")).strip()
        report = data.get("report", {})

        if not question:
            return jsonify({
                "error": "Question is required."
            }), 400

        if not report:
            return jsonify({
                "error": "Analyze expenses first."
            }), 400

        return jsonify({
            "answer": answer_chat(question, report)
        })

    except Exception:
        return jsonify({
            "error": "Unable to answer the question."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
