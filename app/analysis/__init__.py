"""Analysis: finding what is wrong in the data, not just answering what was asked.

Two halves that need each other.  The deterministic half (``diagnose``, ``trends``)
computes facts in SQL -- data-quality defects, period-over-period movement, segment
outliers.  It is exact, costs no model calls, and never invents anything.  The agent
half (``agent``) uses those facts as its starting evidence, probes further with its
own queries, and writes the result up in language.

The split matters: a model asked "where are we lagging" with no evidence will
narrate plausibly and be wrong. Given measured findings it has something to explain.
"""
