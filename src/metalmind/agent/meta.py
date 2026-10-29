import json
import time
import urllib.error
import urllib.request
from metalmind.agent.provider import ProposalResponse,UsageRecord
from metalmind.agent.prompts import build_prompt
from metalmind.agent.candidates import validate_proposal
from metalmind.config import load_api_key

class ProviderError(RuntimeError):
    def __init__(self,message,*,status=None,requests=1):
        super().__init__(message);self.status=status;self.requests=requests

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

class MetaProvider:
    def __init__(self,settings,*,transport=None):
        self.settings=settings;self.transport=transport
        self.key=None if transport else load_api_key()

    def _request(self,body,timeout):
        request=urllib.request.Request("https://api.meta.ai/v1/responses",data=json.dumps(body).encode(),
            headers={"Authorization":"Bearer "+self.key.get_secret_value(),"Content-Type":"application/json"},method="POST")
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request,timeout=timeout) as response: return json.load(response)
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Meta API HTTP {exc.code}",status=exc.code) from None
        except (urllib.error.URLError,TimeoutError):
            raise ProviderError("Meta API connection failed or timed out",status=503) from None

    def propose(self,context):
        body={"model":self.settings.model,"input":build_prompt(context),
            "reasoning":{"effort":self.settings.reasoning_effort},
            "max_output_tokens":min(context.max_output_tokens,self.settings.max_output_tokens)}
        attempts=min(3,context.remaining_requests)
        deadline=time.monotonic()+min(context.timeout_seconds,self.settings.request_timeout_seconds)
        for attempt in range(attempts):
            try:
                remaining=deadline-time.monotonic()
                if remaining<=0: raise ProviderError("Meta request budget elapsed",status=408)
                raw=self.transport(body) if self.transport else self._request(body,remaining)
                break
            except ProviderError as exc:
                exc.requests=attempt+1
                if exc.status not in (429,500,502,503,504) or attempt+1==attempts: raise
                time.sleep(min(2**attempt,max(0,deadline-time.monotonic())))
        usage=raw.get("usage") or {};details=usage.get("output_tokens_details") or {}
        record=UsageRecord(input_tokens=usage.get("input_tokens"),output_tokens=usage.get("output_tokens"),reasoning_tokens=details.get("reasoning_tokens"))
        text="".join(part.get("text","") for item in raw.get("output",[]) if item.get("type")=="message"
            for part in item.get("content",[]) if part.get("type")=="output_text")
        try:
            if raw.get("status")!="completed": raise ValueError("response was incomplete")
            proposal=validate_proposal(text);error=None
        except ValueError as exc: proposal=None;error=str(exc)[:2048]
        return ProposalResponse(proposal=proposal,error=error,model=raw.get("model",self.settings.model),usage=record,requests=attempt+1)
