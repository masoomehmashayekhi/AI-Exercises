import uuid
import json
from datetime import datetime
from .rag_tool import RAGTool
from .web_search import WebSearchTool
import os

TICKETS_FILE = "data/tickets.txt"


class TicketAPISimulator:
    def __init__(self):
        self.tickets = {}
        self.valid_cities = [
            "Tehran", "Mashhad", "Isfahan", "Shiraz", "Tabriz",
            "Kerman", "Rasht", "Ahvaz", "Yazd", "Kish"
        ]

        os.makedirs("data", exist_ok=True)
        if not os.path.exists(TICKETS_FILE):
            open(TICKETS_FILE, "w").close()

        self._load_from_file()

    def _load_from_file(self):
        try:
            with open(TICKETS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ticket = json.loads(line)
                    self.tickets[ticket["ticket_id"]] = ticket
        except Exception as e:
            print("Error loading ticket file:", e)

    def _append_to_file(self, ticket: dict):
        with open(TICKETS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    def _rewrite_file(self): 
        with open(TICKETS_FILE, "w", encoding="utf-8") as f:
            for t in self.tickets.values():
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
 

    def book_ticket(self, data: dict) -> dict:
        try:
            origin = data.get("origin")
            destination = data.get("destination")
            date_str = data.get("date")
            passenger = data.get("passenger")

            if origin not in self.valid_cities or destination not in self.valid_cities:
                return {"error": "Invalid city name", "status": 400}


            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except:
                return {"error": "Invalid date format (YYYY-MM-DD)", "status": 400}

            required = ["full_name", "national_id", "phone"]
            if not passenger or any(k not in passenger for k in required):
                return {"error": "Missing passenger information", "status": 400}

            base_price = 500_000
            distance_factor = abs(len(origin) - len(destination)) * 20_000
            final_price = base_price + distance_factor

            ticket_id = str(uuid.uuid4())

            ticket_info = {
                "ticket_id": ticket_id,
                "origin": origin,
                "destination": destination,
                "travel_date": date_str,
                "passenger": passenger,
                "price": final_price,
                "status": "confirmed"
            }


            self.tickets[ticket_id] = ticket_info
            
            
            self._append_to_file(ticket_info)

            return {
                "status": 200,
                "message": "Ticket booked successfully",
                "data": ticket_info
            }

        except Exception as e:
            return {"error": str(e), "status": 500}



    def cancel_ticket(self, ticket_id: str) -> dict:
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return {"error": "Ticket not found", "status": 404}

        ticket["status"] = "cancelled"
        refund = int(ticket["price"] * 0.7)


        self._rewrite_file()

        return {
            "status": 200,
            "message": "Ticket cancelled",
            "refund_amount": refund,
            "ticket_id": ticket_id
        }
    

    def get_ticket_info(self, ticket_id: str) -> dict:
        ticket = self.tickets.get(ticket_id)

        if not ticket:
            return {"error": "Ticket not found", "status": 404}

        return {"status": 200, "data": ticket}



class Tools:
    def __init__(self, tavily_apikey:str):
        self.ticket_api = TicketAPISimulator()
        self.rag= RAGTool()
        self.rag.clear_collection()
        self.rag.ingest_folder("./data")
        self.web= WebSearchTool(tavily_apikey)

    def run(self, tool_name: str, params: dict):
        if tool_name == "api_book_ticket":
            return self.ticket_api.book_ticket(params)

        if tool_name == "api_cancel_ticket":
            return self.ticket_api.cancel_ticket(params.get("ticket_id"))

        if tool_name == "api_get_ticket_info":
            return self.ticket_api.get_ticket_info(params.get("ticket_id"))
        
        if tool_name == "rag_query":
            q = params.get("query") or params.get("q") or ""
            top_k = int(params.get("top_k", 3))
            return {"results": self.rag.query(q, top_k=top_k)}
        
        if tool_name == "web_search":
            query = params.get("query") or params.get("q") or ""
            location = params.get("location")
            max_results = int(params.get("max_results", 5))
            return self.web.search(query=query, location=location, max_results=max_results)
        return {"error": "Unknown tool", "status": 400}
