"""Stage 4B: sequence-to-teacher-head head-to-head alignment.

Online RiNALMo-micro student with intermediate exits at layers 6/8/12,
one Transformer alignment adapter per preserved teacher head, and direct
teacher-head-to-student-head distillation losses against the slim teacher
attention cache. No teacher consensus aggregation: every teacher head keeps
its model/layer/head identity end to end.
"""
