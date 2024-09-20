#include <SPI.h>
#include <MFRC522.h>
#include <LiquidCrystal.h>

// Initialise the library with the numbers of the interface pins
const int rs = 10, en = 9, d4 = 5, d5 = 6, d6 = 7, d7 = 8;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

#define SS_PIN 53
#define RST_PIN 3
const int BUTTON_PIN = 2;  // Pin where the button is connected
const int RED_PIN = 22;
const int GREEN_PIN = 24;
const int BLUE_PIN = 26;

MFRC522 mfrc522(SS_PIN, RST_PIN); // Create MFRC522 instance.
char receivedData[100];
int dataIndex = 0;
void setup() {
    Serial.begin(250000);  // High baud rate for faster communication
    SPI.begin();
    mfrc522.PCD_Init();
    randomSeed(analogRead(0));

    pinMode(RED_PIN, OUTPUT);
    pinMode(GREEN_PIN, OUTPUT);
    pinMode(BLUE_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    
    lcd.begin(16, 2);
    lcd.setCursor(0, 0);
    lcd.print("Press the button");
    lcd.setCursor(0, 1);
    lcd.print("to start");
}

void loop() {
    if (digitalRead(BUTTON_PIN) == LOW) {  
        // Print a message to the serial monitor:
        Serial.println("START");
        lcd.clear();
        lcd.print("Start scanning!");
        blinkLED(BLUE_PIN, 500);

        // Add a small delay to debounce the button:
        delay(200);  
    }

    // Check if data is available to read from the serial port
    if (Serial.available() > 0) {
        String msg = Serial.readString();

        if (msg == "done"){
            lcd.clear();
            lcd.print("Collection done!");
            }
        }

            
            
    // Look for new RFID cards
    if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial()) {
        return;
    }

    // Blink the blue LED for 500ms to indicate a scan
    blinkLED(BLUE_PIN, 500);

    // Read the UID
    char UID[20];
    readUID(UID);

    // Handle different UID cases
    handleUID(UID);

    delay(500);  // Delay before scanning the next card

}

void readUID(char* UID) {
    UID[0] = '\0';  // Ensure the UID string starts empty

    for (byte i = 0; i < mfrc522.uid.size; i++) {
        char hexValue[4];
        snprintf(hexValue, sizeof(hexValue), "%02X ", mfrc522.uid.uidByte[i]);
        strcat(UID, hexValue);
    }

    UID[strlen(UID) - 1] = '\0';  // Remove the trailing space
}

void handleUID(char* UID) {
    lcd.clear();
    
    if (strcmp(UID, "04 9E F9 55 6F 61 80") == 0) {
        displayData(UID, "Passenger Class:", "1");
    } else if (strcmp(UID, "04 50 22 4E 6F 61 80") == 0) {
        displayData(UID, "Passenger Class:", "2");
    } else if (strcmp(UID, "04 16 AC 55 6F 61 80") == 0) {
        displayData(UID, "Passenger Class:", "3");
    } else if (strcmp(UID, "04 33 25 51 6F 61 80") == 0) {
        displayData(UID, "Sex:", "Male");
    } else if (strcmp(UID, "04 80 F2 6D 8F 61 80") == 0) {
        displayData(UID, "Sex:", "Female");
    } else if (strcmp(UID, "04 61 7F 53 6F 61 80") == 0) {
        displayRandomAge(UID, 0, 2);
    } else if (strcmp(UID, "04 25 C4 4E 6F 61 81") == 0) {
        displayRandomAge(UID, 1, 13);
    } else if (strcmp(UID, "04 8C 3C 53 6F 61 80") == 0) {
        displayRandomAge(UID, 13, 20);
    } else if (strcmp(UID, "04 87 78 56 6F 61 80") == 0) {
        displayRandomAge(UID, 20, 40);
    } else if (strcmp(UID, "04 BA 63 56 6F 61 80") == 0) {
        displayRandomAge(UID, 40, 60);
    } else if (strcmp(UID, "04 A7 9C 4E 6F 61 80") == 0) {
        displayRandomAge(UID, 60, 76);
    } else if (strcmp(UID, "04 49 05 2F 4F 61 80") == 0) {
        displayRandomAge(UID, 75, 80);
    } else if (strcmp(UID, "04 30 C1 55 6F 61 80") == 0) {
        displayData(UID, "Sibling:", "0");
    } else if (strcmp(UID, "04 BA 01 4F 6F 61 80") == 0) {
        displayData(UID, "Sibling:", "1");
    } else if (strcmp(UID, "04 25 48 56 6F 61 81") == 0) {
        displayData(UID, "Siblings:", "2");
    } else if (strcmp(UID, "04 B6 7E 56 6F 61 80") == 0) {
        displayData(UID, "Siblings:", "3");
    } else if (strcmp(UID, "04 5B 76 4E 6F 61 80") == 0) {
        displayData(UID, "Siblings:", "4");
    } else if (strcmp(UID, "04 1B 35 56 6F 61 80") == 0) {
        displayData(UID, "Siblings:", "5");
    } else if (strcmp(UID, "04 33 A2 2E 4F 61 80") == 0) {
        displayData(UID, "Siblings:", "6");
    } else if (strcmp(UID, "04 86 0F 53 6F 61 80") == 0) {
        displayData(UID, "Parent:", "0");
    } else if (strcmp(UID, "04 CD 9A 54 6F 61 80") == 0) {
        displayData(UID, "Parent:", "1");
    } else if (strcmp(UID, "04 CB E6 4E 6F 61 80") == 0) {
        displayData(UID, "Parent:", "2");
    } else if (strcmp(UID, "04 2B AD 65 5F 61 80") == 0) {
        displayData(UID, "Parents:", "3");
    } else if (strcmp(UID, "04 31 11 63 5F 61 80") == 0) {
        displayData(UID, "Parents:", "4");
    } else if (strcmp(UID, "04 64 51 66 5F 61 80") == 0) {
        displayData(UID, "Parents:", "5");
    } else if (strcmp(UID, "04 B7 D2 30 4F 61 81") == 0) {
        displayData(UID, "Parents:", "6");
    } else if (strcmp(UID, "04 43 B4 6E 8F 61 80") == 0) {
        displayFare(UID, 500, 901);
    } else if (strcmp(UID, "04 05 3B 51 6F 61 80") == 0) {
        displayFare(UID, 900, 2001);
    } else if (strcmp(UID, "04 E6 A7 22 CE 11 90") == 0) {
        displayFare(UID, 2000, 3501);
    } else if (strcmp(UID, "04 A0 4D 6E 8F 61 80") == 0) {
        displayFare(UID, 3500, 7000);
    } else if (strcmp(UID, "04 6F C9 22 CE 11 90") == 0) {
        displayFare(UID, 7000, 9501);
    } else if (strcmp(UID, "04 7D C5 22 CE 11 91") == 0) {
        displayFare(UID, 9500, 10000);
    } else if (strcmp(UID, "04 AD 01 22 CE 11 94") == 0) {
        displayCountry(UID, "England");
    } else if (strcmp(UID, "04 2E 64 22 CE 11 91") == 0) {
        displayCountry(UID, "France");
    } else if (strcmp(UID, "04 3F 47 22 CE 11 90") == 0) {
        displayCountry(UID, "Ireland");
    } else if (strcmp(UID, "04 01 64 22 CE 11 91") == 0) {
        displaySurvived(UID, "yes", GREEN_PIN);
    } else if (strcmp(UID, "04 B7 BA 22 CE 11 90") == 0) {
        displaySurvived(UID, "no", RED_PIN);
    }
}

void displayData(String UID, String label, String value) {
    Serial.println(UID + "," + value);
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(label);
    lcd.setCursor(0, 1);
    lcd.print(value);
    blinkBlueLED();
}

void displayRandomAge(String UID, int min, int max) {
    int randomAge = random(min, max + 1); // Ensure max is inclusive
    displayData(UID, "Age:", String(randomAge));
}

void displayFare(String UID, int min, int max) {
    int fare = random(min, max + 1); // Ensure max is inclusive
    Serial.println(UID + "," + String(fare));
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Fare:");
    lcd.setCursor(0, 1);
    lcd.print(fare);
    blinkBlueLED();
}

void displayCountry(String UID, const char* country) {
    Serial.println(String(UID) + "," + country);
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Embarked from:");
    lcd.setCursor(0, 1);
    lcd.print(country);
    blinkBlueLED();
}

void displaySurvived(String UID, const char* survived, int ledPin) {
    Serial.println(String(UID) + "," + survived);
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Survived:");
    lcd.setCursor(0, 1);
    lcd.print(survived);
    digitalWrite(ledPin, HIGH);
    delay(1000);
    digitalWrite(ledPin, LOW);
}

void displayReceivedData(String data) {
    // Assume the data is in the format "survived,not_survived"
    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
        String survived = data.substring(0, commaIndex);
        String notSurvived = data.substring(commaIndex + 1);

        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Surv: ");
        lcd.print(survived);
        lcd.setCursor(0, 1);
        lcd.print("Not Surv: ");
        lcd.print(notSurvived);
    }
}

void blinkLED(int pin, int duration) {
    digitalWrite(pin, HIGH);
    delay(duration);
    digitalWrite(pin, LOW);
}

void blinkBlueLED() {
    blinkLED(BLUE_PIN, 100);
}
