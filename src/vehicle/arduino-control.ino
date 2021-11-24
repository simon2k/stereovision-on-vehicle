int E1 = 5;
int E2 = 6;
int M1 = 4;
int M2 = 7;

void stop() {
  digitalWrite(E1, 0);
  digitalWrite(E2, 0);
}

void moveForward(char speed)
{
  analogWrite (E1, speed);
  digitalWrite(M1, HIGH);

  analogWrite (E2, speed);
  digitalWrite(M2, HIGH);
}

void turnLeft(char speed) {
  analogWrite (E1, speed);
  digitalWrite(M1, LOW);

  analogWrite (E2, speed);
  digitalWrite(M2, HIGH);
}

void turnRight(char speed) {
  analogWrite (E1, speed);
  digitalWrite(M1, HIGH);

  analogWrite (E2, speed);
  digitalWrite(M2, LOW);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10);
  Serial.flush();
  Serial.println("Ready");
}

void loop() {
  if (Serial.available()){
    String command = Serial.readStringUntil('\n');
    command.trim();

    String direction = command.substring(0, 1);
    int speed = command.substring(1, 4).toInt();

    if (direction == "s") {
      stop();
      Serial.println(command);
      return;
    }

    if (direction == "f") {
      moveForward(speed);
    } else if (direction == "r") {
      turnRight(speed);
    } else if (direction == "l") {
      turnLeft(speed);
    }

    Serial.println(command);
  }

  delay(50);
}
