import os
import jinja2

class PromptGenerator:

    def __init__(self, prompt_dir):

        self.env = jinja2.Environment(loader=jinja2.PackageLoader('math_infilling', package_path=prompt_dir))

    def create_qa_prompt(self, question, answer, prompt_name):
        template = self.env.get_template(f"{prompt_name}.txt")

        return template.render(question=question, answer=answer)

    def create_q_prompt(self, question, prompt_name):
        template = self.env.get_template(f"{prompt_name}.txt")

        return template.render(question=question)

    def create_prompt(self, prompt_name, **kwargs):
        template = self.env.get_template(f"{prompt_name}.txt")

        return template.render(**kwargs)

    def create_sr_prompt(self, question, answer, prompt_name, completion_init, completion_fb=None, two=False):
        template = self.env.get_template(f"{prompt_name}.txt")
        if two:
            return template.render(question=question, answer=answer, 
            comp_init=completion_init, comp_fb=completion_fb)

        return template.render(question=question, answer=answer, comp_init= completion_init)


    def create_php_prompt(self, question, answer, prompt_name, prev_ans ):
        template = self.env.get_template(f"{prompt_name}.txt")
        return template.render(question=question, answer=answer, prev_ans=prev_ans)


    

