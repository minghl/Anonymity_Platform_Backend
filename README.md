## 2. setting up for Backend

Users not only need to run the existing frontend, but also need to run the backend interface. To run our pre written Flask PREDANO application from 0 to 1, the following steps need to be followed:

Configure Environment:
Ensure that Python and pip package management tools are installed.
Install Flask framework using pip: 

```
pip install Flask
```

Install depencies packages:

Download some environment packages required to run our backend Flask program from the requirements. txt document

```
pip install −r requirements . txt
```

Running Flask application：
Open the command line and enter the project directory. Execute the  command to start the Flask application.

```
flask run --without-threads
```

Accessing the application:
Open a web browser and enter url Access Flask application. Confirm that the expected response content is displayed in the browser.

```
http://127.0.0.1:3000/
http://localhost:3000/
```

Follow the above steps to successfully run our written Data Anonymity Tool (PREDANO) backend Flask application from 0 to 1.

