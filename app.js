var express = require('express');
var app = express();
var path = require('path');

//Route requirements
//var checkDate = require('./api/routes/checkDate');

app.use('/static', express.static(__dirname + '/static'));
//Setup logging (morgan) and body parsing
app.use(express.urlencoded({extended: false}));
app.use(express.json());

app.get('/', function(request, response) {
    response.sendFile(path.join(__dirname, 'index.html'));
});


//Prevent CORS errors
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', '*');
    if (req.method === 'OPTIONS'){
        res.header('Access-Control-Allow-Methods', 'PUT, POST, PATCH, DELETE, GET, UPDATE');
        return res.status(200).json({});
    }
    next();
});

// Routing
app.get('/', function(request, response) {
    response.sendFile(path.join(__dirname, 'index.html'));
});

//routes that handle requests
//app.use('/api/checkDate', checkDate);

//Handle requests that didn't meet above routes
app.use((req, res, next) => {
    const error = new Error('Not found');
    error.status = 404;
    next(error);
});

app.use((error, req, res, next) => {
    res.status(error.status || 500);
    res.json({
        error: {
            message: error.message
        }
    });
});

module.exports = app;