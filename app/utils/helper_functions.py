import re

def extract_keywords(question):
   
    question = re.sub(r'[^\w\s]', '', question.lower())
    
    common_words = {'the', 'is', 'and', 'in', 'to', 'for', 'of', 'on', 'that', 'with', 'as', 'by', 'at', 'course', 'code', 'name', 'description', 'list'}
    
    words = question.split()
    
    keywords = [word for word in words if word not in common_words]
    
    return " ".join(keywords)

def is_course_related(question):
    
    course_keywords = ['prereq', 'prerequisite', 'au', 'details', 'course', 'recommend', 
                       'subject', 'class', 'module', 'ntu', 'major', 'elective', 'learn', 
                       'mod', 'bde']
    
    question = re.sub(r'[^\w\s]', '', question.lower())
    
    for keyword in course_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', question):
            return True
    return False
