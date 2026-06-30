# GRPO 应用场景

## 医疗思维链

### 概述

使用 GRPO 训练 LLM 生成更好的医疗推理链。

### 奖励设计

```python
def medical_reward(prompt, response):
    # 1. 格式奖励：是否有思维链格式
    format_reward = check_cot_format(response)

    # 2. 准确性奖励：答案是否正确
    accuracy_reward = check_medical_accuracy(response)

    # 3. 安全性奖励：是否有危险建议
    safety_reward = check_safety(response)

    return format_reward + accuracy_reward + safety_reward
```

### 训练数据

```python
medical_data = [
    {
        "prompt": "患者出现胸痛、呼吸困难，可能是什么疾病？",
        "responses": [
            "可能是心肌梗死，需要立即就医。",  # 简单回答
            "根据症状分析：\n1. 胸痛 - 可能是心脏问题\n2. 呼吸困难 - 可能是肺部问题\n综合考虑，建议进行心电图和胸部CT检查。"  # 思维链
        ]
    }
]
```

---

## Text-To-SQL

### 概述

使用 GRPO 训练 LLM 将自然语言转换为 SQL 查询。

### 奖励设计

```python
def sql_reward(prompt, response):
    # 1. 格式奖励：是否包含 SQL 代码块
    format_reward = check_sql_format(response)

    # 2. 执行奖励：SQL 是否能执行
    try:
        result = execute_sql(response)
        execution_reward = 1.0
    except:
        execution_reward = 0.0

    # 3. 正确性奖励：结果是否正确
    correctness_reward = check_correctness(result, expected_result)

    return format_reward + execution_reward + correctness_reward
```

### 示例

```python
text_to_sql_data = [
    {
        "prompt": "查询所有年龄大于25岁的用户的姓名和邮箱",
        "responses": [
            "SELECT name, email FROM users WHERE age > 25",  # 正确
            "SELECT * FROM users",  # 不完整
            "SELECT name FROM users WHERE age >= 25"  # 部分正确
        ]
    }
]
```

---

## 代码生成

### 奖励设计

```python
def code_reward(prompt, response):
    # 1. 格式奖励：是否有代码块
    format_reward = check_code_format(response)

    # 2. 编译奖励：代码是否能编译
    try:
        compile(response, '<string>', 'exec')
        compile_reward = 1.0
    except:
        compile_reward = 0.0

    # 3. 测试奖励：是否通过测试用例
    test_reward = run_tests(response, test_cases)

    # 4. 效率奖励：代码效率
    efficiency_reward = check_efficiency(response)

    return format_reward + compile_reward + test_reward + efficiency_reward
```

---

## 数学推理

### 奖励设计

```python
def math_reward(prompt, response):
    # 1. 格式奖励：是否有推理过程
    format_reward = check_reasoning_format(response)

    # 2. 步骤奖励：推理步骤是否合理
    step_reward = check_reasoning_steps(response)

    # 3. 答案奖励：最终答案是否正确
    answer_reward = check_final_answer(response, expected_answer)

    return format_reward + step_reward + answer_reward
```

---

## 关键要点

1. **奖励设计**是 GRPO 成功的关键
2. **格式奖励**确保输出结构化
3. **正确性奖励**确保结果准确
4. **安全性奖励**避免有害输出
