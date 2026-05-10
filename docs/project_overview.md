# BrandMind Project Overview

## What is BrandMind?
BrandMind is a multi-agent LLM system that converts natural language 
brand descriptions into complete brand identity starter kits including 
font pairings, color palettes, and brand tone/voice.

## Why BrandMind?
Professional brand design is expensive and inaccessible to early-stage 
startups. Existing AI tools generate isolated design elements without 
brand context. BrandMind addresses both gaps.

## How It Works
1. User submits a natural language brand brief
2. Agent 1 classifies archetype and extracts constraints
3. Agent 2 retrieves fonts, colors, and design rules from real databases
4. Agent 3 verifies WCAG compliance, coherence, and constraints
5. If failed, specific feedback triggers revision (max 3 iterations)
6. Final brand kit displayed in Streamlit frontend

## Team
- Sylvia Zhang — Agent 2, LangGraph pipeline, Streamlit frontend
- Helen Wei — Agent 3, baseline benchmark, ablation study
- Fiona Kao — Agent 1, evaluation metrics, human study, GitHub setup

## Course
DS-UA 301: Generative AI — NYU Spring 2026
