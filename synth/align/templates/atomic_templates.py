"""Question and thinking templates for atomic skill QA generation.

Design principle: high diversity in thinking phrasing, consistent answer format.
Templates are composed at generation time by randomly selecting from each pool,
so the conditional probability of any single format is low.
"""

import random
from typing import List, Tuple, Optional

# ============================================================
# Question templates — ~10 per atom type
# ============================================================

GLOBAL_MEAN_QUESTIONS = [
    "What is the mean of the {metric} timeseries?",
    "Calculate the average value of {metric} across the entire series.",
    "What is the overall mean of {metric}?",
    "Find the mean of the {metric} timeseries.",
    "Compute the mean value of {metric}.",
    "What does the average of {metric} come out to?",
    "Determine the mean of the full {metric} series.",
    "Report the mean value of the {metric} data.",
    "What is the average across all data points of {metric}?",
    "What is the mean for {metric} over the entire timeseries?",
    "Find the overall average of {metric}.",
    "What mean value does the {metric} timeseries have?",
]

CHUNKED_MEAN_QUESTIONS = [
    "Compute the mean of each chunk of 16 in {metric} from index {start} to index {end}.",
    "What are the means of consecutive 16-point chunks of {metric} from index {start} to {end}?",
    "Break {metric} from index {start} to {end} into chunks of 16 and compute each chunk's mean.",
    "Find the per-chunk averages (chunk size 16) for {metric} between index {start} and {end}.",
    "For {metric} from index {start} to {end}, divide into chunks of 16 and report each chunk's mean.",
    "Calculate the mean of every 16-point window in {metric} spanning index {start} to {end}.",
    "Partition {metric} from index {start} to {end} into 16-point chunks and compute each mean.",
    "Report the chunk means (chunk size 16) for {metric} over the range [{start}, {end}].",
    "Compute the average for every consecutive block of 16 values in {metric} from index {start} to {end}.",
    "What is the mean of each 16-element segment of {metric} from index {start} to index {end}?",
    "Divide the {metric} data from index {start} to {end} into groups of 16, and find each group's mean.",
    "List the means of non-overlapping 16-point windows in {metric} from index {start} to {end}.",
]

INTERVAL_MEAN_QUESTIONS = [
    "What is the mean of {metric} from index {start} to index {end}?",
    "Calculate the average of {metric} between index {start} and {end}.",
    "Find the mean of {metric} over the range [{start}, {end}].",
    "What is the average value of {metric} from position {start} to {end}?",
    "Compute the mean of {metric} in the interval [{start}, {end}].",
    "Determine the average of {metric} from index {start} to {end}.",
    "What does the mean of {metric} come out to from index {start} to {end}?",
    "Report the mean of {metric} between index {start} and {end}.",
    "What is the overall average of {metric} from index {start} to index {end}?",
    "Find the average value of {metric} across the range [{start}, {end}].",
    "For {metric}, what is the mean from index {start} to {end}?",
    "Compute the average of {metric} over positions {start} through {end}.",
]

# ============================================================
# Thinking template components — composed at generation time
# ============================================================

THINK_OPENERS = [
    "Let me work through this step by step.",
    "I'll compute this by breaking the data into chunks.",
    "Let me analyze this systematically.",
    "I need to compute this step by step.",
    "Let me break this down and compute chunk by chunk.",
    "I'll work through this methodically.",
    "Let me divide the data and compute.",
    "To find this, I'll break the range into chunks of 16.",
    "I'll compute this by dividing into windows of 16.",
    "Let me approach this step by step.",
    "I need to break this into smaller pieces to compute the result.",
    "Let me compute this in stages, using chunks of 16.",
    "I'll go through this chunk by chunk.",
    "Let me think about this carefully, dividing into chunks.",
    "I'll tackle this by chunking the data.",
]

RANGE_DESCRIPTIONS = [
    "The range is [{start}, {end}], containing {n} data points.",
    "Looking at {metric} from index {start} to {end} ({n} values).",
    "The interval [{start}, {end}] has {n} data points.",
    "Working with {n} data points from index {start} to {end}.",
    "I'm looking at {metric} from position {start} to {end}, which is {n} values.",
    "The data from index {start} to {end} spans {n} points.",
    "Considering the range [{start}, {end}]: {n} data points total.",
    "The range covers indices {start} through {end} — {n} values.",
    "{n} data points in the range [{start}, {end}].",
    "From index {start} to {end}, there are {n} data points.",
]

CHUNK_INTROS = [
    "Dividing into chunks of 16: {n_chunks} chunks.",
    "Breaking into {n_chunks} windows of size 16.",
    "This gives {n_chunks} chunks of 16.",
    "Chunking into groups of 16: {n_chunks} chunks.",
    "Splitting into {n_chunks} blocks of 16.",
    "With chunk size 16, I get {n_chunks} chunks.",
    "That makes {n_chunks} chunks of 16 data points each.",
    "I'll divide this into {n_chunks} consecutive chunks of size 16.",
]

# Each style is a function: (chunk_index, start, end, mean_str, partial_note) -> str
CHUNK_LINE_STYLES = {
    'bracket': lambda i, s, e, m, p: f"[{s}-{e}]: mean = {m}{p}",
    'labeled': lambda i, s, e, m, p: f"chunk {i} [{s}-{e}]: mean = {m}{p}",
    'verbose': lambda i, s, e, m, p: f"Chunk {i} (indices {s} to {e}): average = {m}{p}",
    'window': lambda i, s, e, m, p: f"window {i} [{s}-{e}]: {m}{p}",
    'arrow': lambda i, s, e, m, p: f"[{s}-{e}] → {m}{p}",
    'indented': lambda i, s, e, m, p: f"  {i}. [{s}, {e}] mean = {m}{p}",
    'dash': lambda i, s, e, m, p: f"- [{s}-{e}]: {m}{p}",
    'colon': lambda i, s, e, m, p: f"chunk {i}: [{s}-{e}], mean {m}{p}",
}

PARTIAL_CHUNK_NOTES = [
    " ({n} values)",
    " (partial, {n} points)",
    " ({n} data points)",
    " (last chunk: {n} values)",
    " (partial chunk, {n} points)",
    "",  # sometimes omit
    "",
]

AGGREGATION_PHRASES = [
    "Mean of chunk means: {computation} = {result}",
    "Averaging the chunk means: {computation} = {result}",
    "Taking the mean of these: {computation} = {result}",
    "Overall mean = ({sum_str}) / {n} = {result}",
    "Average of chunk means: {computation} = {result}",
    "Computing the mean of these chunk means: {computation} = {result}",
    "Finally, the average: {computation} = {result}",
    "The mean of these values: {computation} = {result}",
]

# ============================================================
# Answer templates (after </think>) — few, consistent
# ============================================================

GLOBAL_MEAN_ANSWERS = [
    "The mean of {metric} is {value}.",
    "For {metric}, the global mean is {value}.",
    "The overall mean of {metric} is {value}.",
]

CHUNKED_MEAN_ANSWERS = [
    "The per-chunk means of {metric} are {value}.",
    "The chunk means for {metric} over that range are {value}.",
    "For {metric}, the chunk-wise means are {value}.",
]

INTERVAL_MEAN_ANSWERS = [
    "The mean of {metric} over that interval is {value}.",
    "For {metric}, the interval mean is {value}.",
    "The average of {metric} in that range is {value}.",
]
