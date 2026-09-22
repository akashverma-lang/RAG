"""Structured-data path: spreadsheets become real SQL tables, queried with SQL.

Top-k retrieval can only ever show the model a handful of rows, so it cannot count,
sum, rank or filter across a whole sheet.  This package loads tabular sheets into
SQLite and answers those questions with generated SQL instead.
"""
