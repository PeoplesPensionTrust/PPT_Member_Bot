This folder houses the training data that would be fed into the 'asset_generation' file to produce the embeddings for the bot to use.

The file the 'asset_generation' script takes is supposed to be a .txt file.

The intents and responses are of the format shown in the example below. This format must be maintained unless a major update to the formatting is undertaken:

# Intent 1: Ask about Occupational Pension Scheme

Entity: Occupational Pension Scheme (OPS)

User: 
What's the Occupational Pension Scheme?

What is OPS?

How does OPS work?

What are the benefits of OPS?


Assistant: 
The Occupational Pension Scheme is a mandatory pension plan for formal sector employees under Tier 2 of the Pension Act. Both you and your employer contribute a percentage of your basic salary, which is deducted through payroll. These contributions are invested to provide a lump sum payout when you retire. You also enjoy tax relief on these contributions, enhancing your retirement benefits. Would you like to know the specific contribution rates or how the funds are invested?


The line "#Intent 1" specifies what action the user wants to take. 

The "Entity" provides the context of what the user wants to find out.

The "User" section and every line of either a question or a command/prompt statement specifies examples of ways the user might ask a question to the bot.

The "Assistant:" section provides the appropriate answer the bot should give in the event that it finds the user query to map to this particular intent.




