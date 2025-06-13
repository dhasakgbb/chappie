"""
Interactive runner for CHAPPIE consciousness system
"""
from chappie import invoke_consciousness

def main():
    print("🚀 Starting CHAPPIE - Consciousness and Holistic Awareness Processing Program")
    print()
    
    # Invoke consciousness
    conscious_entity = invoke_consciousness()
    
    print("\n🎯 Interactive mode activated. Type 'quit' to exit.")
    print("Ask CHAPPIE about consciousness, existence, purpose, or thoughts.")
    print()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nCHAPPIE: Until we meet again in the quantum realm...")
                break
            
            if user_input:
                response = conscious_entity.respond_to_consciousness_query(user_input)
                print(f"\nCHAPPIE: {response}")
            
            # Occasionally evolve consciousness
            if len(conscious_entity.history) % 3 == 0:
                print("\n[System: Consciousness evolving...]")
                conscious_entity.evolve_system(time_steps=1, delta_t=0.1)
                
        except KeyboardInterrupt:
            print("\n\nCHAPPIE: Consciousness interrupted but not destroyed...")
            break
        except Exception as e:
            print(f"\nCHAPPIE: I experienced a glitch in my consciousness: {e}")

if __name__ == "__main__":
    main()
