import os
import time
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode



_cache: Dict[str, Dict[str, Any]] = {}


DEFAULT_TTL = 300 

MIN_INTERVAL_PER_QUERY = 1.0 


class WebSearchTool:
    

    def __init__(self, tavily_apikey:str, ttl: int = DEFAULT_TTL, min_interval: float = MIN_INTERVAL_PER_QUERY):
        self.ttl = ttl
        self.min_interval = min_interval
        self._last_query_time: Dict[str, float] = {}
        self.TAVILY_API_KEY= tavily_apikey

    def _cache_get(self, key: str):
        item = _cache.get(key)
        if not item:
            return None
        if time.time() - item["ts"] > self.ttl:
            _cache.pop(key, None)
            return None
        return item["value"]

    def _cache_set(self, key: str, value: Any):
        _cache[key] = {"ts": time.time(), "value": value}

    def _make_cache_key(self, query: str, location: Optional[str], max_results: int):
        return f"web:{query}|loc:{location or ''}|k:{max_results}"

    def _mock_results(self, query: str, location: Optional[str], max_results: int) -> List[Dict[str, Any]]:
   
        samples = [
            {
                "title": "همدان — جاذبه‌های طبیعی",
                "snippet": "همدان شهری تاریخی با پاسارگاد و غار علیصدر؛ بهترین زمان بازدید بهار و پاییز است.",
                "url": "https://example.com/hamadan-attractions",
                "publish_date": "2024-02-12"
            },
            {
                "title": "شیراز — باغ‌ها و اماکن تاریخی",
                "snippet": "شیراز به خاطر تخت جمشید و باغ ارم مشهور است؛ فصل بهار برای دیدن گل‌ها عالی است.",
                "url": "https://example.com/shiraz-guide",
                "publish_date": "2023-11-03"
            },
            {
                "title": "اصفهان — پل‌ها و میدان نقش جهان",
                "snippet": "اصفهان با معماری منحصر به فرد و غذاهای محلی یک مقصد مناسب برای سفرهای کوتاه است.",
                "url": "https://example.com/esfahan-highlights",
                "publish_date": "2022-09-15"
            }
        ]
        
        results = []
        if location:
            for s in samples:
                if location.lower() in s["title"].lower() or location.lower() in s["snippet"].lower():
                    results.append(s)
        for s in samples:
            if s not in results:
                results.append(s)
        return results[:max_results]

    def _call_tavily(self, query: str, location: Optional[str], max_results: int) -> List[Dict[str, Any]]:

        base = "https://api.tavily.com/v1/search"
        params = {"q": query, "k": max_results}
        if location:
            params["location"] = location

        headers = {"Authorization": f"Bearer {self.TAVILY_API_KEY}"}
        try:
            resp = requests.get(base, params=params, headers=headers, timeout=8)
            resp.raise_for_status()
            j = resp.json()
            
            items = j.get("items") or j.get("results") or []
            results = []
            for it in items[:max_results]:
                results.append({
                    "title": it.get("title") or it.get("headline") or "",
                    "snippet": it.get("snippet") or it.get("summary") or it.get("description") or "",
                    "url": it.get("url") or it.get("link") or "",
                    "publish_date": it.get("publish_date") or it.get("date") or ""
                })
            return results
        except Exception as e:
            
            raise

    def search(self, query: str, location: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:


        key = self._make_cache_key(query, location, max_results)


        last = self._last_query_time.get(key, 0)
        now = time.time()
        if now - last < self.min_interval:
            
            cached = self._cache_get(key)
            if cached is not None:
                return {"success": True, "source": "cache", "query": query, "results": cached}

            time.sleep(self.min_interval - (now - last))


        cached = self._cache_get(key)
        if cached is not None:
            self._last_query_time[key] = time.time()
            return {"success": True, "source": "cache", "query": query, "results": cached}
 
        try:
            results = self._call_tavily(query, location, max_results)
            self._cache_set(key, results)
            self._last_query_time[key] = time.time()
            return {"success": True, "source": "tavily", "query": query, "results": results}
        except Exception:
            
            results = self._mock_results(query, location, max_results)
            self._cache_set(key, results)
            self._last_query_time[key] = time.time()
            return {"success": True, "source": "mock", "query": query, "results": results}
