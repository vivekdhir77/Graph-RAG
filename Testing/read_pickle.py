#!/usr/bin/env python
import pickle

try:
    with open('knowledge_graphs.pkl', 'rb') as f:
        data = pickle.load(f)
        print("Content of knowledge_graphs.pkl:")
        print(data)
except Exception as e:
    print(f"Error: {e}")
