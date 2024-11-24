/*
SQLyog - Free MySQL GUI v5.0
Host - 5.0.45-community-nt : Database - cryptbitcoin
*********************************************************************
Server version : 5.0.45-community-nt
*/


create database if not exists `cryptbitcoin`;

USE `cryptbitcoin`;

/*Table structure for table `agentbuyedtransactions` */

DROP TABLE IF EXISTS `agentbuyedtransactions`;

CREATE TABLE `agentbuyedtransactions` (
  `id` int(11) NOT NULL auto_increment,
  `agentName` varchar(100) NOT NULL,
  `agentemail` varchar(100) NOT NULL,
  `currencyname` varchar(100) NOT NULL,
  `currentprice` float NOT NULL,
  `quantity` int(11) NOT NULL,
  `payableammount` float NOT NULL,
  `cardnumber` varchar(40) NOT NULL,
  `nameoncard` varchar(100) NOT NULL,
  `cardexpiry` varchar(100) NOT NULL,
  `cvv` int(11) NOT NULL,
  `cdate` datetime default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=latin1;

/*Data for the table `agentbuyedtransactions` */

insert into `agentbuyedtransactions` values 
(1,'Tivari Khan','tivarijhadhav23@gmail.com','Bitcoin',8816.93,3,26450.8,'1223565689895652','Mohini','2021-11',568,'2019-10-01 09:38:54'),
(2,'Tivari Khan','tivarijhadhav23@gmail.com','Etherum',178.468,55,9815.74,'3636858596601256','Alex Hales','2023-11',589,'2019-10-01 10:05:57'),
(3,'Tivari Khan','tivarijhadhav23@gmail.com','Bitcoin',8816.93,10,88169.3,'5636454512125858','Sai krishna','2023-03',693,'2019-10-01 10:06:55'),
(4,'Tivari Khan','tivarijhadhav23@gmail.com','Ripple',0.388858,150,58.3287,'8566123589895652','Sagar','2021-01',541,'2019-10-01 10:33:45'),
(5,'Shalini','manaswini234@gmail.com','Bitcoin',8816.93,42,370311,'5891454563632510','Harikrishna','2023-01',584,'2019-10-01 10:36:18'),
(6,'Shalini','manaswini234@gmail.com','Ripple',0.388858,560,217.76,'1253565689852222','Suhasini','2023-01',452,'2019-10-02 09:41:32');

/*Table structure for table `agentregister` */

DROP TABLE IF EXISTS `agentregister`;

CREATE TABLE `agentregister` (
  `id` int(11) NOT NULL auto_increment,
  `email` varchar(100) NOT NULL,
  `pswd` varchar(100) NOT NULL,
  `username` varchar(100) NOT NULL,
  `mobile` varchar(100) NOT NULL,
  `pan` varchar(100) NOT NULL,
  `state` varchar(100) NOT NULL,
  `location` varchar(100) NOT NULL,
  `cryptcurrency` varchar(100) NOT NULL,
  `status` varchar(100) NOT NULL,
  `authkey` varchar(100) NOT NULL,
  `cdate` datetime NOT NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=latin1;

/*Data for the table `agentregister` */

insert into `agentregister` values 
(1,'sivakeshavanna562@gmail.com','siva','Shiva Kethan','9870263635','DNOPA8956H','Telangana','Mahbubnagar','Bitcoin','waiting','waiting','2019-09-28 00:00:00'),
(2,'tivarijhadhav23@gmail.com','tivari','Tivari Khan','9705899636','uoljk8214m','Telangana','Hyderabad','Bitcoin','activated','Ri8o9R92','2019-09-28 00:00:00'),
(3,'niharikaravath@aol.com','niharika','Niharika Sampoorna','970588523','ddche8958p','Telangana','Ranga Reddy','Ethereum','waiting','waiting','2019-09-28 00:00:00'),
(4,'manaswini234@gmail.com','manaswini','Shalini','9705814146','kjoln58996','Telangana','Hyderabad','Bitcoin','activated','l7y3jk0B','2019-09-28 00:00:00'),
(5,'bhargav.podila@gmail.com','1234','bhargav','9581670405','5222007','Andhra Pradesh','Guntur','Bitcoin','activated','OADzgSpU','2019-10-03 08:57:00');

/*Table structure for table `agentscryptoquantity` */

DROP TABLE IF EXISTS `agentscryptoquantity`;

CREATE TABLE `agentscryptoquantity` (
  `currencyname` varchar(100) NOT NULL,
  `useremail` varchar(100) NOT NULL,
  `quantity` int(11) default NULL,
  `id` int(11) NOT NULL auto_increment,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `currencyname` (`currencyname`,`useremail`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=latin1;

/*Data for the table `agentscryptoquantity` */

insert into `agentscryptoquantity` values 
('Bitcoin','tivarijhadhav23@gmail.com',6,1),
('Etherum','tivarijhadhav23@gmail.com',15,2),
('Ripple','tivarijhadhav23@gmail.com',150,3),
('Bitcoin','manaswini234@gmail.com',17,4),
('Ripple','manaswini234@gmail.com',410,5);

/*Table structure for table `blockchainledger` */

DROP TABLE IF EXISTS `blockchainledger`;

CREATE TABLE `blockchainledger` (
  `id` int(11) NOT NULL auto_increment,
  `customeremail` varchar(100) NOT NULL,
  `agentemail` varchar(100) NOT NULL,
  `currencyname` varchar(100) NOT NULL,
  `quantity` int(11) default NULL,
  `paidammout` float NOT NULL,
  `blockchainmoney` float NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;

/*Data for the table `blockchainledger` */

insert into `blockchainledger` values 
(1,'lx160cm@gmail.com','tivarijhadhav23@gmail.com','Bitcoin',2,19661.8,2261.1),
(2,'lx160cm@gmail.com','tivarijhadhav23@gmail.com','Bitcoin',2,19661.8,2261.1),
(3,'lx160cm@gmail.com','tivarijhadhav23@gmail.com','Bitcoin',2,19661.8,2261.1),
(4,'lx160cm@gmail.com','tivarijhadhav23@gmail.com','Bitcoin',1,9830.88,1130.55),
(5,'lx160cm@gmail.com','tivarijhadhav23@gmail.com','Etherum',15,2984.88,343.261),
(6,'nilamsairam@gmail.com','manaswini234@gmail.com','Ripple',150,65.04,7.4796),
(7,'nilamsairam@gmail.com','tivarijhadhav23@gmail.com','Etherum',25,4974.8,572.102),
(8,'nilamsairam@gmail.com','manaswini234@gmail.com','Bitcoin',25,245772,28263.8);

/*Table structure for table `currencychnagetable` */

DROP TABLE IF EXISTS `currencychnagetable`;

CREATE TABLE `currencychnagetable` (
  `id` int(11) NOT NULL auto_increment,
  `currencyname` varchar(100) NOT NULL,
  `conversionRate` float NOT NULL,
  `newCurrencyValue` float NOT NULL,
  `originalCurrencyValue` float NOT NULL,
  `chnageValue` float NOT NULL,
  `profitorloss` varchar(50) NOT NULL,
  `changedate` datetime NOT NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `currencyname` (`currencyname`,`changedate`)
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=latin1;

/*Data for the table `currencychnagetable` */

insert into `currencychnagetable` values 
(1,'Bitcoin',1.5,12228.8,8000,4228.75,'Gain','2019-09-29 00:00:00'),
(2,'Bitcoin',1.03,8397.08,8000,397.075,'Gain','2019-09-28 00:00:00'),
(3,'Bitcoin',1.05,8816.93,8000,816.934,'Gain','2019-09-30 00:00:00'),
(4,'Etherum',1.03,178.468,165,13.4681,'Gain','2019-09-30 00:00:00'),
(22,'Bitcoin',1.5,8917.23,8000,8917.23,'Gain','2019-10-02 00:00:00'),
(23,'Bitcoin',-1.6,8774.55,8000,8774.55,'loss','2019-10-03 00:00:00'),
(24,'Etherum',1.2,183.319,165,183.319,'Gain','2019-10-03 00:00:00'),
(25,'Ripple',1.2,0.393524,0.25,0.393524,'Gain','2019-10-03 00:00:00');

/*Table structure for table `currencyrate` */

DROP TABLE IF EXISTS `currencyrate`;

CREATE TABLE `currencyrate` (
  `currencytype` varchar(100) NOT NULL,
  `doller` float NOT NULL,
  `rupee` float NOT NULL,
  `originalprice` float NOT NULL,
  PRIMARY KEY  (`currencytype`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `currencyrate` */

insert into `currencyrate` values 
('Bitcoin',8774.55,577505,8000),
('Etherum',183.319,12956.1,165),
('Ripple',0.393524,27.6548,0.25);

/*Table structure for table `customercontaincoins` */

DROP TABLE IF EXISTS `customercontaincoins`;

CREATE TABLE `customercontaincoins` (
  `id` int(11) NOT NULL auto_increment,
  `currencyName` varchar(100) NOT NULL,
  `customeremail` varchar(100) NOT NULL,
  `quantity` int(11) default NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `currencyName` (`currencyName`,`customeremail`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=latin1;

/*Data for the table `customercontaincoins` */

insert into `customercontaincoins` values 
(1,'Bitcoin','lx160cm@gmail.com',7),
(2,'Etherum','lx160cm@gmail.com',15),
(3,'Ripple','nilamsairam@gmail.com',150),
(4,'Etherum','nilamsairam@gmail.com',25),
(5,'Bitcoin','nilamsairam@gmail.com',25);

/*Table structure for table `django_session` */

DROP TABLE IF EXISTS `django_session`;

CREATE TABLE `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime NOT NULL,
  PRIMARY KEY  (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `django_session` */

insert into `django_session` values 
('0gwdzb8ap4rj0hxsbl6xuydh72tgh5yq','NDhkY2IxMmQyMmRjYWRlZjc3NTMzZjVlZjEzZmI3YzEyZGUwZmZlYTp7ImlkIjoyLCJsb2dnZWRhZ2VudCI6IlRpdmFyaSBLaGFuIiwiZW1haWwiOiJ0aXZhcmlqaGFkaGF2MjNAZ21haWwuY29tIn0=','2019-10-17 09:10:34'),
('33n6obgf3kyza6ttk3zbrkrkknid69i3','ZmQxZWNkNjJhMzBlYTJjN2FiYTI3OGE5OTI4MjRiOWMxZGIzMGE5Zjp7ImlkIjoxLCJsb2dnZWRhZ2VudCI6IlRpdmFyaSBLaGFuIiwiZW1haWwiOiJseDE2MGNtQGdtYWlsLmNvbSIsImxvZ2dlZHVzZXIiOiJBbGV4IEhhbGVzIn0=','2019-10-17 09:15:15'),
('56wz2ltmyp7fe9qi8u9cwbhbu4jy89ul','YTk2NTQ5YjBmY2U2YWNhZGJjMmEyMjhiMDliYmEzMWI4OTk5MDc1Zjp7ImlkIjoxLCJsb2dnZWR1c2VyIjoiQWxleCBIYWxlcyIsImVtYWlsIjoibHgxNjBjbUBnbWFpbC5jb20ifQ==','2019-10-17 06:26:44'),
('i47t19fuu11w7hewhmfmkjdj9k9221x4','ODBjOGI3MjRhNjYxYjgzNmVlZTlhNTFkMDJlMjJiNzc4Mjk1M2NkOTp7ImlkIjo1LCJsb2dnZWR1c2VyIjoic2FiYXJpIiwiZW1haWwiOiJiaGFyZ2F2LnBvZGlsYUBnbWFpbC5jb20iLCJsb2dnZWRhZ2VudCI6ImJoYXJnYXYifQ==','2019-10-17 08:58:08'),
('y3cmma9ayumuhjjd6r7v7ct98skasys3','YTk2NTQ5YjBmY2U2YWNhZGJjMmEyMjhiMDliYmEzMWI4OTk5MDc1Zjp7ImlkIjoxLCJsb2dnZWR1c2VyIjoiQWxleCBIYWxlcyIsImVtYWlsIjoibHgxNjBjbUBnbWFpbC5jb20ifQ==','2019-10-15 12:00:26');

/*Table structure for table `userbuyingcryptomodel` */

DROP TABLE IF EXISTS `userbuyingcryptomodel`;

CREATE TABLE `userbuyingcryptomodel` (
  `id` int(11) NOT NULL auto_increment,
  `customername` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `currencyname` varchar(100) NOT NULL,
  `quantity` int(11) default NULL,
  `agentemail` varchar(100) NOT NULL,
  `singlecoingamount` float NOT NULL,
  `payableammount` float NOT NULL,
  `cardnumber` varchar(100) NOT NULL,
  `nameoncard` varchar(100) NOT NULL,
  `cardexpiry` varchar(100) NOT NULL,
  `cvv` int(11) default NULL,
  `cdate` datetime default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;

/*Data for the table `userbuyingcryptomodel` */

insert into `userbuyingcryptomodel` values 
(1,'Alex Hales','lx160cm@gmail.com','Bitcoin',2,'tivarijhadhav23@gmail.com',9830.88,19661.8,'5625479363658545','Sonali Mithra','2022-11',562,'2019-10-02 06:36:02'),
(2,'Alex Hales','lx160cm@gmail.com','Bitcoin',2,'tivarijhadhav23@gmail.com',9830.88,19661.8,'5625479363658545','Sonali Mithra','2022-11',562,'2019-10-02 06:49:56'),
(3,'Alex Hales','lx160cm@gmail.com','Bitcoin',2,'tivarijhadhav23@gmail.com',9830.88,19661.8,'5625479363658545','Sonali Mithra','2022-11',562,'2019-10-02 06:50:32'),
(4,'Alex Hales','lx160cm@gmail.com','Bitcoin',1,'tivarijhadhav23@gmail.com',9830.88,9830.88,'9845002269695656','Monisha','2022-12',589,'2019-10-02 07:15:19'),
(5,'Alex Hales','lx160cm@gmail.com','Etherum',15,'tivarijhadhav23@gmail.com',198.99,2984.88,'8989565610102536','Monalisa','2021-02',473,'2019-10-02 07:29:31'),
(6,'Akhilesh','nilamsairam@gmail.com','Ripple',150,'manaswini234@gmail.com',0.43,65.04,'1586458925634578','SusanAilash','2021-01',895,'2019-10-02 09:43:06'),
(7,'Akhilesh','nilamsairam@gmail.com','Etherum',25,'tivarijhadhav23@gmail.com',198.99,4974.8,'5890100125638956','kotireddy','2021-01',589,'2019-10-02 09:43:45'),
(8,'Akhilesh','nilamsairam@gmail.com','Bitcoin',25,'manaswini234@gmail.com',9830.88,245772,'5680125636254589','Mangali','2022-01',589,'2019-10-02 09:44:30');

/*Table structure for table `userregister` */

DROP TABLE IF EXISTS `userregister`;

CREATE TABLE `userregister` (
  `id` int(11) NOT NULL auto_increment,
  `email` varchar(100) NOT NULL,
  `pswd` varchar(100) NOT NULL,
  `username` varchar(100) NOT NULL,
  `mobile` varchar(100) NOT NULL,
  `pan` varchar(100) NOT NULL,
  `state` varchar(100) NOT NULL,
  `location` varchar(100) NOT NULL,
  `status` varchar(100) NOT NULL,
  `authkey` varchar(100) NOT NULL,
  `cdate` datetime NOT NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=latin1;

/*Data for the table `userregister` */

insert into `userregister` values 
(1,'lx160cm@gmail.com','alex','Alex Hales','9849012345','baqpm3931d','Andhra Pradesh','East Godavari','activated','uwiguZ0c','2019-09-27 00:00:00'),
(2,'dkotireddy123@gmail.com','koti','Koti Reddy','9849045612','IRCTC0254L','Telangana','Karimnagar','activated','pHaPYt4m','2019-09-27 00:00:00'),
(3,'thrishula123@gmail.com','alle','Thrishula','9701156563','ATMTJ5896K','Telangana','Hyderabad','waiting','waiting','2019-09-27 00:00:00'),
(4,'ravalireddy8019@gmail.com','ravali','Ravali Reddy','9705289563','ACAMU5901M','Telangana','Hyderabad','waiting','waiting','2019-09-27 00:00:00'),
(5,'arjunreddy141@gmail.com','arjun','Arjun Reddy','9756058963','abncp5622l','Telangana','Karimnagar','waiting','waiting','2019-09-28 00:00:00'),
(6,'nilamsairam@gmail.com','saiswarna','Akhilesh','9752658589','dcbvn9856O','Telangana','Karimnagar','activated','tE4xHn3n','2019-09-28 08:57:46'),
(7,'bhargav.podila@gmail.com','1234','sabari','9581670405','522007','Andhra Pradesh','Guntur','activated','yEizSZO0','2019-10-03 08:37:02');
