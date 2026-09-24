from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """
    Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. An influential figure in popular culture, Swift is known for her autobiographical songwriting and artistic reinventions. She is the highest-grossing live music artist, the wealthiest female musician, and one of the best-selling music artists of all time.

Swift signed with Big Machine Records in 2005 and debuted as a country singer with the albums Taylor Swift (2006) and Fearless (2008). The singles "Teardrops on My Guitar", "Love Story", and "You Belong with Me" found crossover success on country and pop radio formats. Her songs began incorporating stronger rock elements with Speak Now (2010) and electronic styles with Red (2012). She subsequently recalibrated her artistic identity from country to pop with the synth-pop album 1989 (2014), while ensuing media scrutiny inspired the trap-imbued Reputation (2017). Through the 2010s, Swift accumulated the US Billboard Hot 100 number-one singles "We Are Never Ever Getting Back Together", "Shake It Off", "Blank Space", "Bad Blood", and "Look What You Made Me Do".
    """
    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
"""
    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5.5")

    chain = summary_prompt_template | llm  # Creates a callable runnable object via LCEL
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
