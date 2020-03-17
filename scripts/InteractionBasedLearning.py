class InteractionBasedLearning:
    
    print("----------------------------------------------------------------------------------------------------")
    print(
        """
        ################ Interaction-based Learning Statistical Package #################
        ############## © All rights reserved with Professor Shawhwa Lo ##################
        ##### Site: http://stat.columbia.edu/department-directory/name/shaw-hwa-lo/ #####
        """ )
    print("----------------------------------------------------------------------------------------------------")
    print("README:")
    print("This script has the following functions:")
    print(
    """
    (1) iscore(): this function computes the I-score of selected X at predicting Y
    (2) BDA(): this function runs through Backward Dropping Algorithm once
    (3) InteractionLearning(): this function runs many rounds of BDA and 
                               finalize the variables selcted according to I-score
    """ )
    print("ACKNOWLEDGEMENT:")
    print("This script is not-fot-profit and it is a production of my research \nduring my time at Columbia University.")
    print("----------------------------------------------------------------------------------------------------")

    # Define function
    def iscore(X, y):
        # Environment Initiation
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random

        # Create Partition
        partition = X.iloc[:, 0].astype(str)
        if X.shape[1] >= 2:
            partition = partition.astype(str).str.cat(X.iloc[:, 1::].astype(str), sep ="_")
        else:
            partition = partition

        # Local Information
        list_of_partitions = pd.DataFrame(partition.value_counts())
        Pi                 = pd.DataFrame(list_of_partitions.index)
        local_n            = pd.DataFrame(list_of_partitions.iloc[:, :])

        # Compute Influence Score:
        import collections
        n                 = X.shape[0]
        Y_bar             = y.mean()
        grouped           = pd.DataFrame({'y': y, 'X': partition})
        local_mean_vector = pd.DataFrame(grouped.groupby('X').mean())
        local_n           = grouped.groupby('X').count()['y']
        iscore = np.sum(np.array(local_n**2).reshape(1,local_n.shape[0])*np.array([(local_mean_vector['y']-Y_bar)**2]))/np.std(y)/n

        # Output
        return {
            'X': X,
            'y': y,
            'Local Mean Vector': local_mean_vector,
            'Global Mean': Y_bar,
            'Partition': Pi,
            'Number of Samples in Partition': local_n,
            'Influence Score': iscore
        }
    # End of function
    
    # Define function
    def BDA(X, y, num_initial_draw = 4, TYPE=int):
        # Environment Initiation
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        
        # Random Sampling
        newX = X.iloc[:, sorted(random.sample(range(X.shape[1]), num_initial_draw))]

        # BDA
        newX_copy = newX
        iscorePath = []
        selectedX = {}
        for j in range(newX_copy.shape[1]-1):
            unit_scores = []
            for i in range(newX.shape[1]):
                unit_scores.append(InteractionBasedLearning.iscore(
                    X=newX.iloc[:, :].drop([TYPE(newX.columns[i])], axis=1), y=y)['Influence Score'])
                #print(i, unit_scores, np.max(unit_scores), unit_scores.index(max(unit_scores)))
            iscorePath.append(np.max(unit_scores))
            to_drop = unit_scores.index(max(unit_scores))
            newX = newX.iloc[:, :].drop([TYPE(newX.columns[to_drop])], axis=1)
            selectedX[str(j)] = newX

        # Final Output
        finalX = pd.DataFrame(selectedX[str(iscorePath.index(max(iscorePath)))])
        
        # Output
        return {
            'Path': iscorePath,
            'MaxIscore': np.max(iscorePath),
            'newX': finalX,
            'Summary': {
                'Variable Module': np.array(finalX.columns), 
                'Influence Score': np.max(iscorePath) },
            'Brief': [np.array(finalX.columns), [np.max(iscorePath)]] 
        }
    # End of function
    
    # Define function
    def InteractionLearning(newX, y, 
                            testSize=0.3, 
                            num_initial_draw=7, total_rounds=10, top_how_many=3, 
                            nameExists=True, TYPE=int, verbatim=True):
        # Environment Initiation
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        import time
        
        # Split Train and Validate
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=testSize, random_state = 0)
        
        # Start Learning
        start              = time.time()
        listVariableModule = []
        listInfluenceScore = []
        from tqdm import tqdm
        for i in tqdm(range(total_rounds)):
            oneDraw = InteractionBasedLearning.BDA(X=X_train, y=y_train, num_initial_draw=num_initial_draw, TYPE=TYPE)
            listVariableModule.append([np.array(oneDraw['newX'].columns)])
            listInfluenceScore.append(oneDraw['MaxIscore'])
        end = time.time()
        
        # Time Check
        if verbatim == True: 
            print('Time Consumption (in sec):', round(end - start, 2))
            print('Time Consumption (in min):', round((end - start)/60, 2))
            print('Time Consumption (in hr):', round((end - start)/60/60, 2))
        
        # Update Features
        listVariableModule_copy = listVariableModule
        listInfluenceScore_copy = listInfluenceScore
        selectedNom             = listVariableModule[listInfluenceScore.index(np.max(listInfluenceScore))]
        informativeX            = pd.DataFrame(newX[selectedNom[0]])
        listVariableModule_copy = np.delete(listVariableModule_copy, listInfluenceScore_copy.index(np.max(listInfluenceScore)))
        listInfluenceScore_copy = np.delete(listInfluenceScore_copy, listInfluenceScore_copy.index(np.max(listInfluenceScore)))

        for j in range(2, top_how_many):
            selectedNom = listVariableModule_copy[listInfluenceScore_copy.tolist().index(np.max(listInfluenceScore_copy))]
            informativeX = pd.concat([informativeX, pd.DataFrame(newX[selectedNom])], axis=1)
            listVariableModule_copy = np.delete(
                listVariableModule_copy, 
                listInfluenceScore_copy.tolist().index(np.max(listInfluenceScore_copy)))
            listInfluenceScore_copy = np.delete(
                listInfluenceScore_copy, 
                listInfluenceScore_copy.tolist().index(np.max(listInfluenceScore_copy)))
        
        # Generate Brief
        briefResult = pd.DataFrame({'Modules': listVariableModule, 'Score': listInfluenceScore})
        briefResult = briefResult.sort_values(by=['Score'], ascending=False)
        briefResult = briefResult.loc[~briefResult['Score'].duplicated()]
        
        # Generate New Data
        X = newX
        new_X = pd.DataFrame([])
        for ii in range(0, top_how_many):
            # Create Engineered Feature:
            if nameExists:
                X = newX[briefResult.iloc[ii, ][0][0]]
            else:
                X = newX.iloc[:, briefResult.iloc[ii, ][0][0].astype(TYPE)]
            y = y

            # Create Partition
            partition = X.iloc[:, 0].astype(str)
            if X.shape[1] >= 2:
                partition = partition.astype(str).str.cat(X.iloc[:, 1::].astype(str), sep ="_")
            else:
                partition = partition

            # Local Information
            list_of_partitions = pd.DataFrame(partition.value_counts())
            Pi = pd.DataFrame(list_of_partitions.index)
            local_n = pd.DataFrame(list_of_partitions.iloc[:, :])

            # Partition:
            import collections
            n = X.shape[0]
            Y_bar = y.mean()
            grouped = pd.DataFrame({'y': y, 'X': partition})
            local_mean_vector = pd.DataFrame(grouped.groupby('X').mean())

            # Engineered Feature:
            engineeredX = []
            for i in range(len(X)):
                engineeredX.append(local_mean_vector.iloc[partition.iloc[i, ] == local_mean_vector.index, ].iloc[0, 0])

            if nameExists:
                df1 = newX[briefResult.iloc[ii, ][0][0]]
            else: 
                df1 = newX.iloc[:, briefResult.iloc[ii, ][0][0].astype(TYPE)]
            df1.reset_index(drop=True, inplace=True)
            df2 = pd.DataFrame(engineeredX)
            df2.reset_index(drop=True, inplace=True)

            # Concatenate:
            new_X = pd.concat([
                new_X, 
                df1,
                df2
            ], axis=1)


        # Output
        return {
            'List of Variable Modules': listVariableModule,
            'List of Influence Measures': listInfluenceScore,
            'Brief': briefResult,
            'New Features': informativeX,
            'New Data': new_X
        }
    # End of function