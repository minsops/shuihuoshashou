从给定的岗位 JD 和候选人简历中提取：
1. job_title：岗位名称。JD 中明确写出的岗位名优先；没有则根据职责概括一个，不超过 12 字。
2. candidate_name：候选人姓名。简历开头、基本信息或落款的真实姓名优先；找不到则返回 "候选人"。

只输出严格 JSON，匹配 SetupExtraction schema：
{"job_title":"...","candidate_name":"..."}
