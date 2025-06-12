from database.connector import DatabaseConnector
# from generator.base_query_generator import QueryGenerator
from generator.add_query_generator import QueryGenerator
import json

def main():
    # 初始化模块
    db = DatabaseConnector()
    db.save_schema_json()
    
    generator = QueryGenerator("schema.json")
    db_schema = db.extract_schema()
    # print('db_schema:', db_schema)
    
    dataset = []
    identity_id = 0  # 唯一标识计数器

    for i in range(200):
        queries = generator.generate_queries(db_schema)
        print(f'queries {i}:', queries)
        
        for q in queries:
            # 重构数据结构
            formatted_data = {
                "conversations": [
                    {
                        "from": "user",
                        "value": q["question"].strip()
                    },
                    {
                        "from": "assistant",
                        "value": "cot:"+q["cot"].strip()+"\n"+"sql:"+q["sql"].strip() 
                    }
                ]
            }
            dataset.append(formatted_data)
            identity_id += 1  # 标识自增
    
    # 写入文件（确保中文输出）
    with open("text2sql_dataset.json", 'w', encoding='utf-8') as f:  
        json.dump(
            dataset, 
            f, 
            indent=2,
            ensure_ascii=False  # 禁用ASCII转义
        )
    # with open("text2sql_dataset_add.json", 'w', encoding='utf-8') as f:  
    #     json.dump(
    #         dataset, 
    #         f, 
    #         indent=2,
    #         ensure_ascii=False  # 禁用ASCII转义
    #     )

if __name__ == "__main__":
    main()