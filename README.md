This example contains two modal apps.

 1. Our own privately owned LLM.
 2. Our Solara based UI that calls or own private LLM.

# LLM

## Deploy the LLM

```
modal deploy modal_llm.py

```

# Solara UI

## Develop on the UI locally


```
solara run solara_chat.py
```


## Deploy the Solara app to Modal

```
modal deploy solara_chat.py
```
