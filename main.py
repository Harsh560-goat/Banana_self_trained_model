from predict import predict_single, predict_batch
import os
def main():
    print("\n" + "="*60)
    print("IMAGE CLASSIFIER - PREDICTION MENU")
    print("="*60)
    print("1. Predict single image")
    print("2. Predict all images in a folder")
    print("3. Exit")
    print("="*60)
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        image_path = os.path.join(input("Enter image path: "))
        result = predict_single(image_path)
        print(f"Prediction : {result}")
    elif choice == '2':
        folder_path = input("Enter folder path: ")
        results = predict_batch(folder_path)
        print(f"Predictions: {results}")
    elif choice == '3':
        print("Exiting...")
        return
    else:
        print("Invalid choice")
    
    another = input("\nMake another prediction? (y/n): ")
    if another.lower() == 'y':
        main()

if __name__ == "__main__":
    main()
