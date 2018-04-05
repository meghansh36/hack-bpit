const {mongoose} = require('../db/mongoose')

const {Patient} = require('../models/patient')
const {Donor} = require('../models/donor')
const express = require('express')

var app = express()

function bloodGroup(bg) {
    if(bg == 'O'){
        return ['O', 'A', 'B', 'AB']
    }
    if(bg == 'A'){
        return ['A', 'AB']
    }
    if(bg == 'B'){
        return ['B','AB']
    }
    if(bg == 'AB'){
        return ['AB']
    }
}

var organArray = ['heart','kidney', 'liver']

function potential (req,res,next){
console.log(req.body)
Donor.findOne({
    _id: req.body.donor_id,
    "organs.organ_name" : {$all: organArray} 

}).then((organInfo) => {
    if(!organInfo)
    {console.log('checking checking')}

    return organInfo;
}).then((organInfo) => {
    console.log(organInfo)
    Patient.find({
        organ_needed: {
            $in: organArray
        }, //donors
        blood_group: {
            $in: bloodGroup(organInfo.blood_group)
        }
    }).then((data) => {
            var sorted_severity = data.sort((a,b) => {
                return b.severity - a.severity
            });
             console.log(sorted_severity)
             req.potential = sorted_severity
             next()
         });
}).catch((e) => {
    console.log(e)
});

}

module.exports = {potential}

// Patient.find({
//     organ_needed: donor_organ
// }).then((data) => {
//     console.log(data)
// });