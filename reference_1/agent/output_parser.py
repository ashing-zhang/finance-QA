import re
from typing import Dict, Tuple

import json


class OutputParser:

    def parse_response(self, response):
        raise NotImplementedError

'''
    适配如下格式的输出：
        <|startofthink|>
        {
            "api_name": "get_weather",
            "parameters": {
                "city": "Beijing"
            }
        }
        <|endofthink|>

'''
class BSOutputParser(OutputParser):

    def __init__(self):
        super().__init__()

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if '<|startofthink|>' not in response or '<|endofthink|>' not in response:
            return None, None
        try:
            # use regular expression to get result
            re_pattern1 = re.compile(
                pattern=r'<\|startofthink\|>([\s\S]+)<\|endofthink\|>')
            think_content = re_pattern1.search(response).group(1)

            re_pattern2 = re.compile(r'{[\s\S]+}')
            think_content = re_pattern2.search(think_content).group()

            json_content = json.loads(think_content.replace('\n', '').replace('""','"'))
            action = json_content.get('api_name',
                                      json_content.get('name', 'unknown')).strip(' ')
            parameters = json_content.get('parameters', {})

            return action, parameters

        except Exception as e:
            return None, None

'''
    适配如下格式的输出：
        Action: get_weather
        Action Input: {
            "city": "Shanghai"
        }
'''
class QwenOutputParser(OutputParser):

    def __init__(self):
        super().__init__()

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if 'Action' not in response or 'Action Input:' not in response:
            return None, None
        try:
            # use regular expression to get result
            re_pattern1 = re.compile(
                pattern=r'Action:([\s\S]+)Action Input:([\s\S]+)')
            res = re_pattern1.search(response)
            action = res.group(1).strip()
            action_para = res.group(2)

            parameters = json.loads(action_para.replace('\n', ''))

            print(response)
            print(action, parameters)
            return action, parameters
        except Exception:
            return None, None
