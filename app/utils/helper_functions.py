import re

def extract_keywords(question):
    """
    从用户的问题中提取关键词，用于优化向量检索。使用正则表达式匹配名词或关键词。
    """
    # 将问题转换为小写并去除标点符号
    question = re.sub(r'[^\w\s]', '', question.lower())
    
    # 定义关键词列表 (可以根据需要扩展)
    common_words = {'the', 'is', 'and', 'in', 'to', 'for', 'of', 'on', 'that', 'with', 'as', 'by', 'at', 'course', 'code', 'name', 'description', 'list'}
    
    # 分词
    words = question.split()
    
    # 过滤掉常见词汇，只保留有效关键词
    keywords = [word for word in words if word not in common_words]
    
    return " ".join(keywords)

def is_course_related(question):
    """
    判断问题是否与课程相关，使用正则表达式匹配关键的课程相关词汇。
    """
    # 定义课程相关的关键词列表
    course_keywords = ['prereq', 'prerequisite', 'au', 'details', 'course', 'recommend', 
                       'subject', 'class', 'module', 'ntu', 'major', 'elective', 'learn', 
                       'mod', 'bde']
    
    # 将问题转换为小写并去除标点符号
    question = re.sub(r'[^\w\s]', '', question.lower())
    
    # 检查问题中是否包含课程相关的关键词
    for keyword in course_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', question):
            return True
    return False
