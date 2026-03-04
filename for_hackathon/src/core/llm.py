import asyncio
import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("langchain_openai not available, LLM calls will fail")

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, usage_logger: Optional[Callable] = None):
        self.usage_logger = usage_logger
        self.default_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.default_api_key:
            logger.warning("OPENAI_API_KEY not set in environment")
    
    async def call(
        self,
        module: str,
        model_cfg: dict,
        messages: List[BaseMessage],
        ctx: dict,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        response_format: Optional[type] = None,
        timeout: float = 300.0
    ) -> Dict:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain_openai is required for LLM calls")
        
        model_name = model_cfg.get("name", "gpt-4.1")
        temperature = model_cfg.get("temperature", 0.7)
        max_tokens = model_cfg.get("max_tokens", 1000)
        base_url = model_cfg.get("base_url")
        api_key = model_cfg.get("api_key")
        if api_key is None:
            api_key = self.default_api_key
        
        if not base_url and not api_key:
            raise ValueError(
                "For OpenAI API: OPENAI_API_KEY must be set in environment, or api_key must be provided in model config. "
                "For OpenAI-compatible API (vLLM): base_url must be provided in model config."
            )
        
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if base_url:
            llm_kwargs["base_url"] = base_url
            llm_kwargs["extra_body"] = {
                "repetition_penalty": 1.2
            }
            logger.info(f"Using OpenAI-compatible API: base_url={base_url}")
            if api_key:
                llm_kwargs["api_key"] = api_key
        else:
            if not api_key:
                raise ValueError("api_key is required for OpenAI API")
            llm_kwargs["api_key"] = api_key
            logger.info(f"Using OpenAI API: model={model_name}")
        
        llm = ChatOpenAI(**llm_kwargs)
        
        if response_format:
            try:
                llm = llm.with_structured_output(response_format, method="function_calling")
            except Exception as e:
                logger.warning(f"Failed to use structured output with function_calling, falling back to JSON parsing: {e}")
                response_format = None
        
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                try:
                    response = await asyncio.wait_for(
                        llm.ainvoke(messages),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"LLM call timed out after {timeout} seconds")
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                if response_format:
                    if hasattr(response, 'model_dump'):
                        data = response.model_dump()
                        text = json.dumps(data, ensure_ascii=False)
                        logger.debug(f"Structured output (Pydantic): {text[:200]}")
                    elif isinstance(response, dict):
                        text = json.dumps(response, ensure_ascii=False)
                        logger.debug(f"Structured output (dict): {text[:200]}")
                    else:
                        try:
                            data = dict(response) if hasattr(response, '__dict__') else response
                            text = json.dumps(data, ensure_ascii=False)
                            logger.debug(f"Structured output (fallback): {text[:200]}")
                        except Exception as e:
                            logger.warning(f"Failed to convert structured output to JSON: {e}")
                            text = str(response)
                else:
                    if hasattr(response, 'content'):
                        text = response.content
                    elif isinstance(response, str):
                        text = response
                    else:
                        text = str(response)
                    
                    if not text or (isinstance(text, str) and not text.strip()):
                        logger.warning(
                            f"Empty response detected. Response type: {type(response)}, "
                            f"has content attr: {hasattr(response, 'content')}, "
                            f"content value: {repr(getattr(response, 'content', None))}"
                        )
                        if attempt < max_retries - 1:
                            raise ValueError("LLM returned empty response")
                        else:
                            logger.error(
                                "LLM returned empty response on final attempt; "
                                "continuing with empty text instead of failing."
                            )
                            text = ""
                    else:
                        logger.debug(f"Text response: {text[:200]}")
                
                usage_info = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0
                }
                
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    if 'token_usage' in metadata:
                        token_usage = metadata['token_usage']
                        usage_info = {
                            "prompt_tokens": token_usage.get("prompt_tokens", 0),
                            "completion_tokens": token_usage.get("completion_tokens", 0),
                            "total_tokens": token_usage.get("total_tokens", 0),
                            "cached_tokens": token_usage.get("cached_tokens", 0)
                        }
                
                if self.usage_logger:
                    record = {
                        "module": module,
                        "model": model_name,
                        "base_url": base_url if base_url else None,
                        "iter_idx": ctx.get("iter_idx"),
                        "split": ctx.get("split"),
                        "dialogue_id": ctx.get("dialogue_id"),
                        "turn_idx": ctx.get("turn_idx"),
                        "judge_name": ctx.get("judge_name"),
                        "prompt_tokens": usage_info["prompt_tokens"],
                        "completion_tokens": usage_info["completion_tokens"],
                        "total_tokens": usage_info["total_tokens"],
                        "cached_tokens": usage_info["cached_tokens"],
                        "latency_ms": latency_ms,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                    try:
                        self.usage_logger(record)
                    except Exception as e:
                        logger.error(f"Failed to log usage: {e}")
                
                return {
                    "text": text,
                    "usage": {
                        **usage_info,
                        "latency_ms": latency_ms
                    }
                }
                
            except (TimeoutError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"LLM call timed out (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    delay = retry_delay * (3 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"LLM call timed out after {max_retries} attempts")
                    raise
                
            except ValueError as e:
                if "empty response" in str(e).lower():
                    last_error = e
                    
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"LLM returned empty response (attempt {attempt + 1}/{max_retries}), retrying..."
                        )
                        delay = retry_delay * (3 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"LLM returned empty response after {max_retries} attempts")
                        raise
                else:
                    raise
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    delay = retry_delay * (3 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts")
                    raise
        
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")
    
    async def get_embeddings(
        self,
        texts: List[str],
        model_cfg: dict,
        ctx: dict,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        timeout: float = 300.0
    ) -> Dict:
        import httpx
        
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        model_name = model_cfg.get("name", "text-embedding-ada-002")
        base_url = model_cfg.get("base_url")
        api_key = model_cfg.get("api_key")
        if api_key is None:
            api_key = self.default_api_key
        
        if not base_url and not api_key:
            raise ValueError(
                "For OpenAI API: OPENAI_API_KEY must be set in environment, or api_key must be provided in model config. "
                "For OpenAI-compatible API: base_url must be provided in model config."
            )
        
        # Determine API URL
        if base_url:
            # For OpenAI-compatible API, use base_url directly
            api_url = f"{base_url.rstrip('/')}/embeddings"
            headers = {
                "Content-Type": "application/json"
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            # For OpenAI API
            api_url = "https://api.openai.com/v1/embeddings"
            if not api_key:
                raise ValueError("api_key is required for OpenAI API")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                request_data = {
                    "model": model_name,
                    "input": texts
                }
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    try:
                        response = await asyncio.wait_for(
                            client.post(api_url, json=request_data, headers=headers),
                            timeout=timeout
                        )
                        response.raise_for_status()
                        result = response.json()
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Embeddings request timed out after {timeout} seconds")
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Extract embeddings and usage
                if "data" not in result:
                    raise ValueError("Response does not contain 'data' field")
                
                embeddings = [item["embedding"] for item in result["data"]]
                
                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"Number of embeddings ({len(embeddings)}) does not match "
                        f"number of input texts ({len(texts)})"
                    )
                
                usage_info = result.get("usage", {})
                
                if self.usage_logger:
                    record = {
                        "module": ctx.get("module", "embeddings"),
                        "model": model_name,
                        "base_url": base_url if base_url else None,
                        "prompt_tokens": usage_info.get("prompt_tokens", 0),
                        "completion_tokens": 0,
                        "total_tokens": usage_info.get("total_tokens", 0),
                        "cached_tokens": 0,
                        "latency_ms": latency_ms,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                    try:
                        self.usage_logger(record)
                    except Exception as e:
                        logger.error(f"Failed to log usage: {e}")
                
                return {
                    "embeddings": embeddings,
                    "usage": usage_info
                }
                
            except (TimeoutError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"Embeddings request timed out (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    delay = retry_delay * (3 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Embeddings request timed out after {max_retries} attempts")
                    raise
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embeddings request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    delay = retry_delay * (3 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Embeddings request failed after {max_retries} attempts")
                    raise
        
        raise RuntimeError(f"Embeddings request failed after {max_retries} attempts: {last_error}")
